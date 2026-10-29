package com.example.textfromimage

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.TorchState
import androidx.camera.core.ZoomState
import androidx.camera.mlkit.vision.MlKitAnalyzer
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.example.textfromimage.ui.theme.TextFromImageTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.Executors
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TextFromImageTheme {
                TextReaderApp()
            }
        }
    }
}

private data class OcrBlock(
    val text: String,
    val averageLineHeight: Float,
    val lineCount: Int
)

private data class OcrDocument(
    val rawText: String = "",
    val blocks: List<OcrBlock> = emptyList()
)

private data class PreviewTextBox(
    val text: String,
    val bounds: RectF,
    val averageLineHeight: Float,
    val lineCount: Int,
    val alpha: Float = 1f
)

private data class TrackedPreviewTextBox(
    val box: PreviewTextBox,
    val lastSeenMillis: Long
)

private class TextOverlaySmoother {
    private var tracked = emptyList<TrackedPreviewTextBox>()

    fun update(detections: List<PreviewTextBox>, nowMillis: Long): List<PreviewTextBox> {
        val matched = BooleanArray(tracked.size)
        val next = mutableListOf<TrackedPreviewTextBox>()
        detections.forEach { detection ->
            val matchIndex = tracked.indices
                .filterNot { matched[it] }
                .maxByOrNull { index -> matchScore(tracked[index].box, detection) }
                ?.takeIf { index ->
                    tracked[index].box.text == detection.text || intersectionOverUnion(tracked[index].box.bounds, detection.bounds) >= 0.08f
                }
            val stable = if (matchIndex != null) {
                matched[matchIndex] = true
                val previous = tracked[matchIndex].box
                val dx = previous.bounds.centerX() - detection.bounds.centerX()
                val dy = previous.bounds.centerY() - detection.bounds.centerY()
                val movement = kotlin.math.hypot(dx, dy)
                val boxSize = maxOf(previous.bounds.width(), previous.bounds.height()).coerceAtLeast(1f)
                val blend = if (movement < boxSize * 0.18f) 0.16f else 0.30f
                detection.copy(
                    bounds = RectF(
                        lerp(previous.bounds.left, detection.bounds.left, blend),
                        lerp(previous.bounds.top, detection.bounds.top, blend),
                        lerp(previous.bounds.right, detection.bounds.right, blend),
                        lerp(previous.bounds.bottom, detection.bounds.bottom, blend)
                    ),
                    averageLineHeight = lerp(previous.averageLineHeight, detection.averageLineHeight, blend)
                )
            } else detection
            next += TrackedPreviewTextBox(stable, nowMillis)
        }
        tracked.forEachIndexed { index, previous ->
            if (!matched[index]) {
                val age = nowMillis - previous.lastSeenMillis
                if (age in 0..350) {
                    next += previous.copy(box = previous.box.copy(alpha = 1f - age / 350f))
                }
            }
        }
        tracked = next
        return next.map { it.box }
    }

    private fun matchScore(first: PreviewTextBox, second: PreviewTextBox): Float {
        val iou = intersectionOverUnion(first.bounds, second.bounds)
        return if (sameRecognizedText(first.text, second.text)) iou + 1f else iou
    }

    private fun intersectionOverUnion(first: RectF, second: RectF): Float {
        val left = maxOf(first.left, second.left)
        val top = maxOf(first.top, second.top)
        val right = minOf(first.right, second.right)
        val bottom = minOf(first.bottom, second.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val union = first.width() * first.height() + second.width() * second.height() - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun lerp(start: Float, end: Float, fraction: Float) = start + (end - start) * fraction
}

private class ProminentTextTracker {
    private var selected: PreviewTextBox? = null
    private var challenger: PreviewTextBox? = null
    private var challengerSince = 0L
    private var missingSince = 0L

    fun select(boxes: List<PreviewTextBox>, nowMillis: Long): PreviewTextBox? {
        val candidates = boxes.filter { it.text.isNotBlank() && it.alpha >= 0.45f }
        val best = prominentPreviewBox(candidates)
        if (selected == null) {
            selected = best
            missingSince = 0L
            return selected
        }

        val current = candidates.maxByOrNull { candidate -> trackingScore(selected!!, candidate) }
            ?.takeIf { trackingScore(selected!!, it) >= 1f }
        if (current == null) {
            if (missingSince == 0L) missingSince = nowMillis
            if (nowMillis - missingSince < 450L) return selected
            selected = best
            challenger = null
            challengerSince = 0L
            missingSince = 0L
            return selected
        }

        selected = current
        missingSince = 0L
        if (best == null || trackingScore(current, best) >= 1f || best.averageLineHeight < current.averageLineHeight * 1.38f) {
            challenger = null
            challengerSince = 0L
            return selected
        }

        if (challenger == null || trackingScore(challenger!!, best) < 1f) {
            challenger = best
            challengerSince = nowMillis
        } else {
            challenger = best
        }
        if (nowMillis - challengerSince >= 550L) {
            selected = best
            challenger = null
            challengerSince = 0L
        }
        return selected
    }
}

private fun sameRecognizedText(first: String, second: String): Boolean {
    val a = first.lowercase(Locale.ROOT).filter(Char::isLetterOrDigit)
    val b = second.lowercase(Locale.ROOT).filter(Char::isLetterOrDigit)
    if (a == b) return true
    val shorter = minOf(a.length, b.length)
    return shorter >= 4 && (a.contains(b) || b.contains(a))
}

private fun trackingScore(first: PreviewTextBox, second: PreviewTextBox): Float {
    val overlap = intersectionOverUnion(first.bounds, second.bounds)
    return if (sameRecognizedText(first.text, second.text)) overlap + 1f else overlap
}

private fun intersectionOverUnion(first: RectF, second: RectF): Float {
    val left = maxOf(first.left, second.left)
    val top = maxOf(first.top, second.top)
    val right = minOf(first.right, second.right)
    val bottom = minOf(first.bottom, second.bottom)
    val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
    val union = first.width() * first.height() + second.width() * second.height() - intersection
    return if (union > 0f) intersection / union else 0f
}

private fun Text.toOcrDocument(): OcrDocument = OcrDocument(
    rawText = text,
    blocks = textBlocks.map { block ->
        val lineHeights = block.lines.mapNotNull { it.boundingBox?.height()?.toFloat() }
        val averageHeight = if (lineHeights.isNotEmpty()) {
            lineHeights.average().toFloat()
        } else {
            block.boundingBox?.height()?.toFloat()?.div(block.lines.size.coerceAtLeast(1)) ?: 0f
        }
        OcrBlock(block.text, averageHeight, block.lines.size)
    }
)

private fun OcrDocument.prominentBlock(): OcrBlock? = blocks
    .filter { it.text.isNotBlank() }
    .maxWithOrNull(compareBy<OcrBlock> { it.averageLineHeight }.thenBy { -it.text.length })

private fun OcrDocument.formattedText(): String {
    if (blocks.isEmpty()) return rawText
    val lineHeights = blocks.map { it.averageLineHeight }.filter { it > 0f }.sorted()
    val medianHeight = lineHeights.getOrNull(lineHeights.size / 2) ?: 0f
    return blocks.joinToString("\n\n") { block ->
        val text = block.text.trim()
        val looksLikeHeading = medianHeight > 0f &&
            block.averageLineHeight >= medianHeight * 1.35f &&
            block.lineCount <= 2 &&
            text.length <= 120 &&
            '\n' !in text
        if (looksLikeHeading) "\u0000$text" else text
    }
}

@Composable
private fun TextReaderApp() {
    val context = LocalContext.current
    val recognizer = remember { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }
    var document by remember { mutableStateOf(OcrDocument()) }
    var busy by remember { mutableStateOf(false) }
    var scanning by remember { mutableStateOf(false) }
    var paused by rememberSaveable { mutableStateOf(false) }
    var lockedText by rememberSaveable { mutableStateOf<String?>(null) }
    var ttsReady by remember { mutableStateOf(false) }
    var speaking by remember { mutableStateOf(false) }
    var ttsEngine by remember { mutableStateOf<TextToSpeech?>(null) }
    var message by remember { mutableStateOf("") }
    var showOutlines by rememberSaveable { mutableStateOf(true) }
    var focusProminentText by rememberSaveable { mutableStateOf(false) }
    var cleanFormatting by rememberSaveable { mutableStateOf(true) }
    var settingsExpanded by rememberSaveable { mutableStateOf(false) }

    DisposableEffect(recognizer) {
        onDispose { recognizer.close() }
    }

    DisposableEffect(context) {
        val mainHandler = Handler(Looper.getMainLooper())
        val engine = TextToSpeech(context.applicationContext) { status ->
            mainHandler.post { ttsReady = status == TextToSpeech.SUCCESS }
        }
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                mainHandler.post { speaking = true }
            }

            override fun onDone(utteranceId: String?) {
                mainHandler.post { speaking = false }
            }

            override fun onError(utteranceId: String?) {
                mainHandler.post { speaking = false }
            }
        })
        ttsEngine = engine
        onDispose {
            engine.stop()
            engine.shutdown()
            ttsEngine = null
            ttsReady = false
            speaking = false
        }
    }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scanning = false
        paused = false
        lockedText = null
        busy = true
        message = ""
        val image = runCatching { InputImage.fromFilePath(context, uri) }.getOrElse {
            busy = false
            message = "That image could not be opened. Try another one."
            return@rememberLauncherForActivityResult
        }
        recognizer.process(image)
            .addOnSuccessListener { result ->
                document = result.toOcrDocument()
                message = if (result.text.isBlank()) "No text found in this image." else ""
            }
            .addOnFailureListener { message = "Text recognition failed. Try another image." }
            .addOnCompleteListener { busy = false }
    }

    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        scanning = granted
        message = if (granted) "" else "Camera access is needed to scan live. You can still choose an image."
    }

    val prominentBlock = document.prominentBlock()
    val focusedText = lockedText ?: if (focusProminentText) prominentBlock?.text else null
    val outputText = when {
        focusedText != null -> focusedText
        cleanFormatting -> document.formattedText().replace("\u0000", "")
        else -> document.rawText
    }
    val outputDisplay = when {
        outputText.isBlank() -> buildAnnotatedString { append("Your recognized text will appear here.") }
        focusedText != null -> buildAnnotatedString {
            withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp)) { append(outputText) }
        }
        cleanFormatting -> buildAnnotatedString {
            document.formattedText().split("\n\n").forEachIndexed { index, paragraph ->
                if (index > 0) append("\n\n")
                if (paragraph.startsWith("\u0000")) {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp)) {
                        append(paragraph.removePrefix("\u0000"))
                    }
                } else {
                    append(paragraph)
                }
            }
        }
        else -> buildAnnotatedString { append(outputText) }
    }

    Scaffold { insets ->
        Column(
            modifier = Modifier.fillMaxSize().padding(insets).verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                Box(
                    modifier = Modifier.size(52.dp).clip(CircleShape).background(Color(0xFF142A3A)),
                    contentAlignment = Alignment.Center
                ) {
                    Image(
                        painter = painterResource(com.example.textfromimage.R.drawable.ic_launcher_foreground),
                        contentDescription = null,
                        modifier = Modifier.size(46.dp)
                    )
                }
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("TEXTLENS", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                    Text("Read anything", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                    Text("On-device text scanning · works offline", style = MaterialTheme.typography.bodySmall)
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                Button(
                    onClick = { imagePicker.launch("image/*") },
                    modifier = Modifier.weight(1f),
                    enabled = !busy
                ) {
                    Text("Choose image")
                }
                OutlinedButton(
                    onClick = {
                        if (scanning) {
                            scanning = false
                            paused = false
                        } else if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                            paused = false
                            scanning = true
                        }
                        else cameraPermission.launch(Manifest.permission.CAMERA)
                    },
                    modifier = Modifier.weight(1f)
                ) { Text(if (scanning) "Stop scan" else "Live scan") }
            }

            if (scanning) {
                LiveCameraScanner(
                    recognizer = recognizer,
                    showOutlines = showOutlines,
                    paused = paused,
                    onPauseToggle = { paused = !paused },
                    lockedText = lockedText,
                    onBlockTap = { selectedText ->
                        lockedText = if (lockedText == selectedText) null else selectedText
                        focusProminentText = true
                    },
                    onDocument = { detected ->
                        if (detected.rawText.isNotBlank()) {
                            document = detected
                            message = ""
                        }
                    },
                    modifier = Modifier.fillMaxWidth().height(390.dp).clip(RoundedCornerShape(24.dp))
                )
                Text("Tap text to lock focus. Tap it again to return to auto.", style = MaterialTheme.typography.bodySmall)
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text("Preferences", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                            Text("Highlight, focus, and text layout", style = MaterialTheme.typography.bodySmall)
                        }
                        TextButton(onClick = { settingsExpanded = !settingsExpanded }) {
                            Text(if (settingsExpanded) "Done" else "Adjust")
                        }
                    }
                    if (settingsExpanded) {
                        if (scanning) OptionToggle("Highlight selected text", showOutlines) { showOutlines = it }
                        OptionToggle("Focus on prominent text", focusProminentText) {
                            focusProminentText = it
                            if (!it) lockedText = null
                        }
                        OptionToggle("Clean formatting", cleanFormatting) { cleanFormatting = it }
                    }
                }
            }

            if (busy) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator()
                    Text("Reading text…")
                }
            }
            if (message.isNotBlank()) {
                Text(
                    message,
                    color = if (message == "Copied to clipboard.") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall
                )
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Recognized text", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                            Text(
                                if (focusedText != null) "Focused selection" else "Full result",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.secondary
                            )
                        }
                    }
                    SelectionContainer {
                        Text(
                            outputDisplay,
                            style = MaterialTheme.typography.bodyLarge,
                            modifier = Modifier.fillMaxWidth().heightIn(max = 260.dp).verticalScroll(rememberScrollState())
                        )
                    }
                    if (document.rawText.isNotBlank()) {
                        Row {
                            val clipboard = context.getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                            TextButton(onClick = {
                                clipboard.setPrimaryClip(android.content.ClipData.newPlainText("Detected text", outputText))
                                message = "Copied to clipboard."
                            }) { Text(if (focusedText != null) "Copy focused" else "Copy") }
                            TextButton(
                                enabled = ttsReady,
                                onClick = {
                                    if (speaking) {
                                        ttsEngine?.stop()
                                        speaking = false
                                    } else {
                                        val status = ttsEngine?.speak(outputText, TextToSpeech.QUEUE_FLUSH, null, "recognized-text")
                                        if (status == TextToSpeech.ERROR) message = "Speech playback could not start."
                                    }
                                }
                            ) { Text(if (speaking) "Stop reading" else "Read aloud") }
                            TextButton(onClick = { document = OcrDocument(); lockedText = null; message = "" }) { Text("Clear") }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun OptionToggle(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(label, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun LiveCameraScanner(
    recognizer: com.google.mlkit.vision.text.TextRecognizer,
    showOutlines: Boolean,
    paused: Boolean,
    onPauseToggle: () -> Unit,
    lockedText: String?,
    onBlockTap: (String) -> Unit,
    onDocument: (OcrDocument) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val smoother = remember { TextOverlaySmoother() }
    val prominenceTracker = remember { ProminentTextTracker() }
    val cameraController = remember(context) {
        LifecycleCameraController(context).apply {
            cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
            setImageAnalysisBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        }
    }
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            controller = cameraController
        }
    }
    var overlayBoxes by remember { mutableStateOf<List<PreviewTextBox>>(emptyList()) }
    var autoSelectedBox by remember { mutableStateOf<PreviewTextBox?>(null) }
    var zoomRatio by remember { mutableFloatStateOf(1f) }
    var minimumZoom by remember { mutableFloatStateOf(1f) }
    var maximumZoom by remember { mutableFloatStateOf(1f) }
    var hasFlash by remember { mutableStateOf(false) }
    var flashlightOn by remember { mutableStateOf(false) }
    val analyzer = remember(recognizer, mainExecutor) {
        MlKitAnalyzer(
            listOf(recognizer),
            ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED,
            mainExecutor
        ) { result ->
            val text = result.getValue(recognizer)
            if (text != null) {
                if (text.text.isNotBlank()) onDocument(text.toOcrDocument())
                val detections = text.textBlocks.mapNotNull { block ->
                    val bounds = block.boundingBox ?: return@mapNotNull null
                    val heights = block.lines.mapNotNull { it.boundingBox?.height()?.toFloat() }
                    val lineHeight = heights.average().toFloat().takeIf { it.isFinite() && it > 0f }
                        ?: (bounds.height().toFloat() / block.lines.size.coerceAtLeast(1))
                    PreviewTextBox(block.text, RectF(bounds), lineHeight, block.lines.size)
                }
                val stabilized = smoother.update(detections, SystemClock.elapsedRealtime())
                overlayBoxes = stabilized
                autoSelectedBox = prominenceTracker.select(stabilized, SystemClock.elapsedRealtime())
            } else {
                val stabilized = smoother.update(emptyList(), SystemClock.elapsedRealtime())
                overlayBoxes = stabilized
                autoSelectedBox = prominenceTracker.select(stabilized, SystemClock.elapsedRealtime())
            }
        }
    }

    DisposableEffect(cameraController, lifecycleOwner, executor, analyzer) {
        cameraController.setImageAnalysisAnalyzer(executor, analyzer)
        cameraController.bindToLifecycle(lifecycleOwner)
        onDispose {
            cameraController.clearImageAnalysisAnalyzer()
            cameraController.unbind()
            executor.shutdown()
        }
    }

    DisposableEffect(cameraController, lifecycleOwner) {
        val zoomObserver = Observer<ZoomState> { state ->
            zoomRatio = state.zoomRatio
            minimumZoom = state.minZoomRatio
            maximumZoom = state.maxZoomRatio
            hasFlash = cameraController.cameraInfo?.hasFlashUnit() == true
        }
        val torchObserver = Observer<Int> { state -> flashlightOn = state == TorchState.ON }
        cameraController.zoomState.observe(lifecycleOwner, zoomObserver)
        cameraController.torchState.observe(lifecycleOwner, torchObserver)
        onDispose {
            cameraController.zoomState.removeObserver(zoomObserver)
            cameraController.torchState.removeObserver(torchObserver)
        }
    }

    androidx.compose.runtime.LaunchedEffect(paused, cameraController, analyzer) {
        if (paused) cameraController.clearImageAnalysisAnalyzer()
        else cameraController.setImageAnalysisAnalyzer(executor, analyzer)
    }

    Column(modifier = modifier) {
        BoxWithConstraints(modifier = Modifier.fillMaxWidth().weight(1f)) {
            AndroidView(
                factory = { previewView.apply { controller = cameraController } },
                modifier = Modifier.fillMaxSize()
            )
            val automaticBox = autoSelectedBox ?: prominentPreviewBox(overlayBoxes)
            val selectedBox = lockedText?.let { pinned -> overlayBoxes.firstOrNull { sameRecognizedText(it.text, pinned) } }
                ?: automaticBox
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
                    .pointerInput(overlayBoxes, lockedText) {
                        detectTapGestures { tap ->
                            val tappedBox = overlayBoxes.lastOrNull { it.bounds.contains(tap.x, tap.y) }
                            if (tappedBox != null) onBlockTap(tappedBox.text)
                            runCatching {
                                val point = previewView.meteringPointFactory.createPoint(tap.x, tap.y)
                                cameraController.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(point).build())
                            }
                        }
                    }
            ) {
                if (showOutlines && selectedBox != null) {
                    val margin = 8.dp.toPx()
                    val left = (selectedBox.bounds.left - margin).coerceIn(0f, size.width)
                    val top = (selectedBox.bounds.top - margin).coerceIn(0f, size.height)
                    val right = (selectedBox.bounds.right + margin).coerceIn(0f, size.width)
                    val bottom = (selectedBox.bounds.bottom + margin).coerceIn(0f, size.height)
                    val frameWidth = right - left
                    val frameHeight = bottom - top
                    if (frameWidth > 0f && frameHeight > 0f) {
                        drawRect(Color.Black.copy(alpha = 0.58f))
                        drawRoundRect(
                            color = Color.Transparent,
                            topLeft = Offset(left, top),
                            size = Size(frameWidth, frameHeight),
                            cornerRadius = CornerRadius(12.dp.toPx()),
                            blendMode = BlendMode.Clear
                        )
                        val cornerLength = 20.dp.toPx().coerceAtMost(minOf(frameWidth, frameHeight) * 0.35f)
                        val strokeWidth = 3.5.dp.toPx()
                        val corners = listOf(
                            Triple(Offset(left, top), 1f, 1f),
                            Triple(Offset(right, top), -1f, 1f),
                            Triple(Offset(left, bottom), 1f, -1f),
                            Triple(Offset(right, bottom), -1f, -1f)
                        )
                        corners.forEach { (point, horizontal, vertical) ->
                            val color = Color.White.copy(alpha = selectedBox.alpha)
                            drawLine(color, point, Offset(point.x + horizontal * cornerLength, point.y), strokeWidth, cap = StrokeCap.Round)
                            drawLine(color, point, Offset(point.x, point.y + vertical * cornerLength), strokeWidth, cap = StrokeCap.Round)
                        }
                    }
                }
            }
            androidx.compose.material3.Surface(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 12.dp)
                    .width(76.dp)
                    .height(34.dp)
                    .clickable(onClick = onPauseToggle),
                color = Color(0xFF25272B),
                contentColor = Color.White,
                shape = androidx.compose.foundation.shape.RoundedCornerShape(50),
                shadowElevation = 5.dp
            ) {
                Box(contentAlignment = Alignment.Center, modifier = Modifier.padding(horizontal = 14.dp)) {
                    Text(if (paused) "Resume" else "Pause", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                }
            }
        }
        if (maximumZoom > minimumZoom) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("${"%.1f".format(Locale.getDefault(), zoomRatio)}×", style = MaterialTheme.typography.labelLarge)
                androidx.compose.material3.Slider(
                    value = zoomRatio.coerceIn(minimumZoom, maximumZoom),
                    onValueChange = { ratio ->
                        zoomRatio = ratio
                        cameraController.setZoomRatio(ratio)
                    },
                    valueRange = minimumZoom..maximumZoom,
                    modifier = Modifier.weight(1f)
                )
            }
        }
        if (hasFlash) OptionToggle("Flashlight", flashlightOn) { cameraController.enableTorch(it) }
    }
}

private fun prominentPreviewBox(boxes: List<PreviewTextBox>): PreviewTextBox? = boxes
    .filter { it.text.isNotBlank() }
    .maxWithOrNull(compareBy<PreviewTextBox> { it.averageLineHeight }.thenBy { -it.text.length })

