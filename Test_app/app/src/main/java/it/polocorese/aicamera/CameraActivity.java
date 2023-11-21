package it.polocorese.aicamera;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Size;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.codemonkeylabs.fpslibrary.TinyDancer;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class CameraActivity extends AppCompatActivity {

    private static final int REQUEST_PERMISSION_CAMERA = 777;
    private static final int PERMISSION_REQUEST_CODE = 200;
    private ListenableFuture<ProcessCameraProvider> cameraProviderF;
    private ImageAnalysis imageAnalysis;
    private static final int REQUEST_CAMERA = 1001;
    private static final int ANALYSIS_DELAY_MS = 5_000;
    private static final int INVALID_TIME = -1;
    private long lastAnalysisTime = INVALID_TIME;
    private Executor executor = Executors.newSingleThreadExecutor();
    private CustomImageLabelerOptions customImageLabelerOptions;
    PreviewView previewView;
    TextView Label;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera);
        previewView = findViewById(R.id.Camera_photo);
        Label = findViewById(R.id.Camera_Label);
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);

        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
        }

        LocalModel localModel =
                new LocalModel.Builder()
                        .setAssetFilePath("mobilenetv2_050.tflite")
                        .build();


        TinyDancer.create().show(this);

        TinyDancer.create()
                .redFlagPercentage(.1f) // set red indicator for 10%
                .startingGravity(Gravity.TOP)
                .startingXPosition(100)
                .startingYPosition(10)
                .show(this);


        customImageLabelerOptions =
                new CustomImageLabelerOptions.Builder(localModel)
                        .setConfidenceThreshold(0.9f)
                        .setMaxResultCount(3)
                        .build();

        cameraProviderF = ProcessCameraProvider.getInstance(getApplicationContext());

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
        else
            initSource();
    }


    private void startCameraX(@NonNull  ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        Preview preview = new Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        ImageLabeler labeler = ImageLabeling.getClient(customImageLabelerOptions);


        ImageCapture imageCapture =
                new ImageCapture.Builder()
                        .setTargetRotation(previewView.getDisplay().getRotation())
                        .build();

        imageCapture.takePicture(executor, new ImageCapture.OnImageCapturedCallback() {

            @Override
            public void onCaptureSuccess (ImageProxy imageProxy) {
                // Use the image, then make sure to close it before returning from the method
                imageProxy.close();
            }

            @Override
            public void onError (ImageCaptureException exception) {
                // Handle the exception however you'd like
            }
        });

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                final long now = SystemClock.uptimeMillis();
                if (lastAnalysisTime != INVALID_TIME && (now - lastAnalysisTime < ANALYSIS_DELAY_MS)) {
                    lastAnalysisTime = now;
                    imageProxy.close();
                }
                else {
                    Bitmap InputMap = toBitmap(imageProxy.getImage());
                    InputImage inputImage = InputImage.fromBitmap(InputMap,0);

//                    try {
//                        String path = Environment.getExternalStorageDirectory().toString();
//                        OutputStream fOut = null;
//                        Integer counter = 0;
//                        File file = new File(path, "test"+counter+".jpg"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
//                        fOut = new FileOutputStream(file);
//
//
//                        InputMap.compress(Bitmap.CompressFormat.JPEG, 100, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
//                        fOut.close(); // do not forget to close the stream
//
//                        MediaStore.Images.Media.insertImage(getContentResolver(),file.getAbsolutePath(),file.getName(),file.getName());
//                        Label.setText("Done");
//                    } catch (FileNotFoundException e) {
//                        Label.setText(e.getMessage());
//                    } catch (IOException e) {
//                        Label.setText(e.getMessage());
//                        throw new RuntimeException(e);
//                    }

                    labeler.process(inputImage).addOnSuccessListener(imageLabels -> {
                        String message = "Label:\n";
                        message += " - ";
                        int lbl = 1;
                        for (ImageLabel label : imageLabels) {
                            String getLabel = label.getText();
                            String test =  "None";

                            if (getLabel.compareTo(test) != 0) {
                                lbl = 0;
                                message += getLabel + " (" + (Math.round(label.getConfidence() * 100000) / 1000) + "%)" + " ";
                            }
                        }
                        if (lbl == 1)
                            message = "Image is not related";
                        Label.setText(message);
                    });

                    imageProxy.close();


                }


            }
        });
        cameraProvider.bindToLifecycle(this, cameraSelector,imageAnalysis,imageCapture, preview);

    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initSource();
        }

    }


    private void initSource() {
        cameraProviderF.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderF.get();
                startCameraX(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(getApplicationContext()));
    }
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:

                try {
                    ProcessCameraProvider cameraProvider = cameraProviderF.get();
                    cameraProvider.unbindAll();
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                finish();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }



    void toRGB(int[] rgb, byte[] yuv420sp, int width, int height) {

        final int frameSize = width * height;

        for (int j = 0, yp = 0; j < height; j++) {
            int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
            for (int i = 0; i < width; i++, yp++) {
                int y = (0xff & ((int) yuv420sp[yp])) - 16;
                if (y < 0)
                    y = 0;
                if ((i & 1) == 0) {
                    v = (0xff & yuv420sp[uvp++]) - 128;
                    u = (0xff & yuv420sp[uvp++]) - 128;
                }

                int y1192 = 1192 * y;
                int r = (y1192 + 1634 * v);
                int g = (y1192 - 833 * v - 400 * u);
                int b = (y1192 + 2066 * u);

                if (r < 0) r = 0;
                else if (r > 262143) r = 262143;
                if (g < 0) g = 0;
                else if (g > 262143) g = 262143;
                if (b < 0) b = 0;
                else if (b > 262143) b = 262143;

                rgb[yp] = 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
            }
        }
    }
    private Bitmap toBitmap(Image image) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Invalid image format");
        }

        int width = image.getWidth();
        int height = image.getHeight();

        // Order of U/V channel guaranteed, read more:
        // https://developer.android.com/reference/android/graphics/ImageFormat#YUV_420_888
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        ByteBuffer yBuffer = yPlane.getBuffer();
        ByteBuffer uBuffer = uPlane.getBuffer();
        ByteBuffer vBuffer = vPlane.getBuffer();

        // Full size Y channel and quarter size U+V channels.
        int numPixels = (int) (width * height * 1.5f);
        byte[] nv21 = new byte[numPixels];
        int index = 0;

        // Copy Y channel.
        int yRowStride = yPlane.getRowStride();
        int yPixelStride = yPlane.getPixelStride();
        for(int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                nv21[index++] = yBuffer.get(y * yRowStride + x * yPixelStride);
            }
        }

        // Copy VU data; NV21 format is expected to have YYYYVU packaging.
        // The U/V planes are guaranteed to have the same row stride and pixel stride.
        int uvRowStride = uPlane.getRowStride();
        int uvPixelStride = uPlane.getPixelStride();
        int uvWidth = width / 2;
        int uvHeight = height / 2;

        for(int y = 0; y < uvHeight; ++y) {
            for (int x = 0; x < uvWidth; ++x) {
                int bufferIndex = (y * uvRowStride) + (x * uvPixelStride);
                // V channel.
                nv21[index++] = vBuffer.get(bufferIndex);
                // U channel.
                nv21[index++] = uBuffer.get(bufferIndex);
            }
        }


//        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, /* strides= */ null);
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);
//        byte[] imageBytes = out.toByteArray();

        int[] pixels = new int[width * height];
        toRGB(pixels,nv21,width,height);
        Bitmap bmp = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);

        return bmp;
    }


    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }
}
