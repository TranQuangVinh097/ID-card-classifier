package com.example.face_detector.faceDetector;

import static android.content.ContentValues.TAG;

import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.example.face_detector.camera.GraphicOverlay;
import com.example.face_detector.camera.imageAnalyzer;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.util.Date;
import java.util.List;

public class faceDetector extends imageAnalyzer<List<Face>> {

    private FaceDetectorOptions realTimeOpts = new FaceDetectorOptions.Builder()
                                                   .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                                                   .enableTracking()
                                                   .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                                                   .build();

    private FaceDetector detector = FaceDetection.getClient(realTimeOpts);
    public TextView rps_tv;
    public int count = 0;
    public Date start;
    public Date end;

    public faceDetector(GraphicOverlay view, TextView tv) {
        getOverlay(view);
        rps_tv = tv;
    }
    @Override
    protected Task<List<Face>> detectInImage(InputImage inputImage) {
        start = new Date();

        Task<List<Face>> result = detector.process(inputImage).addOnSuccessListener(
                new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        Date end = new Date();
                        Float processedTime = (float)(end.getTime() - start.getTime());
                        rps_tv.setText("RPS: " + (int)(1000 / processedTime) + "");
                    }
                });

        return result;
    }

    @Override
    protected void stop() {
        detector.close();
    }

    @Override
    protected void onSuccess(List<Face> result, GraphicOverlay graphicOverlay, Rect rect) {
         graphicOverlay.clear();
         for (Face face : result) {
             for (FaceLandmark landmark : face.getAllLandmarks()) {
                 PointF inputPoint = landmark.getPosition();
                 RectF inputRect = new RectF();
                 inputRect.set(inputPoint.x, inputPoint.y, inputPoint.x, inputPoint.y);
                 GraphicOverlay.Graphic landmarkG = new faceGraphic(graphicOverlay, inputRect, rect);
                 graphicOverlay.add(landmarkG);
             }
             GraphicOverlay.Graphic faceG = new faceGraphic(graphicOverlay, face, rect, face.getHeadEulerAngleX(), face.getHeadEulerAngleY());
             graphicOverlay.add(faceG);
         }
         graphicOverlay.postInvalidate();
    }

    @Override
    protected void onFailure(Exception e) {
        Log.w(TAG, "onFailure: " + e);
    }

}
