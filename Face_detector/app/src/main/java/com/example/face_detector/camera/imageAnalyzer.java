package com.example.face_detector.camera;

import android.annotation.SuppressLint;
import android.graphics.Rect;
import android.media.Image;
import android.util.Log;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;

public abstract class imageAnalyzer<T> implements ImageAnalysis.Analyzer {

    private GraphicOverlay graphicOverlay;

    protected void getOverlay(GraphicOverlay view) {
        graphicOverlay = view;
    }
    @Override @SuppressLint("UnsafeOptInUsageError")
    public void analyze(ImageProxy imageProxy) {
        Image mediaImage = imageProxy.getImage();
        InputImage inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());

        detectInImage(inputImage).addOnSuccessListener(
            result -> {
                onSuccess(result, graphicOverlay, mediaImage.getCropRect());
                imageProxy.close();
        }).addOnFailureListener(failure -> {
                onFailure(failure);
                imageProxy.close();
            }
        );
    }

    protected abstract Task<T> detectInImage(InputImage inputImage);
    protected abstract void onSuccess(T result, GraphicOverlay graphicOverlay, Rect rect);
    protected abstract void stop();
    protected abstract void onFailure(Exception e);

}
