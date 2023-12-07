package com.example.face_detector.camera;

import android.content.Context;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.example.face_detector.faceDetector.faceDetector;
public class CameraManager {
    private ListenableFuture<ProcessCameraProvider> cameraProviderF;
    private Preview preview;
    private GraphicOverlay graphicOverlay;
    private ImageAnalysis imageAnalysis;
    private ProcessCameraProvider cameraProvider;
    private int cameraSelectorOption = CameraSelector.LENS_FACING_FRONT;
    public TextView aps_tv;
    private LifecycleOwner lifecycleCamera;
    private ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    public CameraManager(Context context, LifecycleOwner lifecycleOwner, PreviewView previewView, GraphicOverlay cameraOverlay, TextView tv) {
        cameraProviderF = ProcessCameraProvider.getInstance(context);
        graphicOverlay = cameraOverlay;
        aps_tv = tv;
        cameraProviderF.addListener(() -> {
            try {
                cameraProvider = cameraProviderF.get();
                lifecycleCamera = lifecycleOwner;
                preview = new Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, selectAnalyzer(graphicOverlay));


                startCameraX();
            } catch (ExecutionException | InterruptedException e) {
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private void startCameraX() {
        cameraProvider.unbindAll();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(cameraSelectorOption)
                .build();


        cameraProvider.bindToLifecycle(lifecycleCamera, cameraSelector,imageAnalysis, preview);

    }

    public void setCameraLen() {
        cameraProvider.unbindAll();

        cameraSelectorOption = (cameraSelectorOption == CameraSelector.LENS_FACING_FRONT)
                                ? CameraSelector.LENS_FACING_BACK
                                : CameraSelector.LENS_FACING_FRONT;

        graphicOverlay.toggleSelector();

        startCameraX();
    }

    private ImageAnalysis.Analyzer selectAnalyzer(GraphicOverlay view) {
            return new faceDetector(view, aps_tv);
    }
}
