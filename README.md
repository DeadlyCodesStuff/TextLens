# TextLens

My phone doesn't have this feature, and I got tired of using websites to pull text out of pictures. So I made an app for it.

TextLens can read text from an image or scan it live with your camera. The recognition runs on your device, so it works offline and your images don't need to be uploaded anywhere.

## What it does

- Pick an image and get its text.
- Point the camera at something and read text as it appears.
- Highlight detected text in the live view, pause the scan, or tap a text block to keep it in focus.
- Copy the result, select part of it, or have it read aloud.
- Tidy up the detected text and focus on the most prominent section.

## Run it

Open this folder in Android Studio and let Gradle sync. Then run the `app` configuration on a device or emulator. Live scanning needs camera permission; picking an image does not.

You can also build a debug APK from this folder:

```sh
./gradlew :app:assembleDebug
```

The APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

TextLens is an early project, so expect a few rough edges. The current Android application ID is `com.example.textfromimage` and should be changed before publishing a release.
