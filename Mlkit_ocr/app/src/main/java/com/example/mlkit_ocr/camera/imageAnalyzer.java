package com.example.mlkit_ocr.camera;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Rect;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.example.mlkit_ocr.MainActivity;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;

public abstract class imageAnalyzer<T> implements ImageAnalysis.Analyzer {

    private GraphicOverlay graphicOverlay;
    private long lastAnalysisTime = SystemClock.uptimeMillis();
    private static final int INVALID_TIME = -1;
    private static final int ANALYSIS_DELAY_MS = 2_00;

    protected void getOverlay(GraphicOverlay view) {
        graphicOverlay = view;
    }
    @Override @SuppressLint("UnsafeOptInUsageError")
    public void analyze(ImageProxy imageProxy) {
        Image mediaImage = imageProxy.getImage();
        InputImage inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
        final long now = SystemClock.uptimeMillis();
        Log.i("TIME", "Now: " + now + " Old: " + ANALYSIS_DELAY_MS);

        if (lastAnalysisTime != INVALID_TIME && (now - lastAnalysisTime < ANALYSIS_DELAY_MS)) {
            imageProxy.close();
        } else {
            lastAnalysisTime = now;
            detectInImage(inputImage).addOnSuccessListener(
                    result -> {
                        onSuccess(result, graphicOverlay, mediaImage.getCropRect());
                        imageProxy.close();
                    }).addOnFailureListener(failure -> {
                        onFailure(failure);
                        imageProxy.close();
                    });
            }

    }

    protected abstract Task<T> detectInImage(InputImage inputImage);
    protected abstract void onSuccess(T result, GraphicOverlay graphicOverlay, Rect rect);
    protected abstract void stop();
    protected abstract void onFailure(Exception e);

}
