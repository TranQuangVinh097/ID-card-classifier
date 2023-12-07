package com.example.face_detector.faceDetector;

import android.content.Context;
import android.content.pm.LauncherApps;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.face_detector.R;
import com.example.face_detector.camera.GraphicOverlay;
import com.example.face_detector.camera.GraphicOverlay.Graphic;
import com.google.mlkit.vision.face.FaceLandmark;
import com.google.mlkit.vision.face.Face;

import java.util.List;

public class faceGraphic extends GraphicOverlay.Graphic{
    private Paint facePositionPaint = new Paint();
    private Paint textPaint = new Paint();
    private Paint boxPaint = new Paint();
    private Rect imgRect;
    private RectF inputFace = new RectF();
    private GraphicOverlay inputOverlay;
    private String angle = "";

    public faceGraphic(GraphicOverlay overlay, RectF landmark, Rect imageRect) {
        int selectedColor = Color.YELLOW;
        getContext(overlay);
        facePositionPaint.setColor(selectedColor);
        boxPaint.setColor(selectedColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(10.0f);

        inputOverlay = overlay;
        imgRect = imageRect;

        inputFace.set(landmark);
    }

    public faceGraphic(GraphicOverlay overlay, Face face, Rect imageRect, Float angleX, Float angleY) {
        int selectedColor = Color.YELLOW;
        getContext(overlay);
        facePositionPaint.setColor(selectedColor);
        int textColor = ContextCompat.getColor(overlay.getContext(), R.color.dark_blue);
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(20);
        boxPaint.setColor(selectedColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5.0f);

        inputOverlay = overlay;
        imgRect = imageRect;

        angle = "X: " + String.format("%.2f", angleX) + " Y: " + String.format("%.2f", angleY);
        inputFace.set(face.getBoundingBox());
    }

    @Override
    protected void draw(@NonNull  Canvas canvas) {
        Rect rect = new Rect();
        calculateRect(inputOverlay,
                      (float)imgRect.height(),
                      (float)imgRect.width(),
                      inputFace).round(rect);
        if (angle != "") {
            canvas.drawText(angle, Math.min(rect.left, rect.right), rect.top - 3, textPaint);
        }
        canvas.drawRect(rect, boxPaint);
    }


}
