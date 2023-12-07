package com.example.mlkit_ocr.textDectector;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;

import androidx.annotation.NonNull;

import com.example.mlkit_ocr.camera.GraphicOverlay;
import com.google.mlkit.vision.text.Text;

public class viewGraphic extends GraphicOverlay.Graphic{
    private Paint facePositionPaint = new Paint();
    private Paint textPaint = new Paint();
    private Paint boxPaint = new Paint();
    private Rect imgRect;
    private RectF inputBox = new RectF();
    private GraphicOverlay inputOverlay;
    private String inputText = "";

    public viewGraphic(GraphicOverlay overlay, Text.TextBlock block, Rect imageRect) {
        int selectedColor = Color.YELLOW;
        getContext(overlay);
        facePositionPaint.setColor(selectedColor);
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(20);
        boxPaint.setColor(selectedColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5.0f);

        inputOverlay = overlay;
        imgRect = imageRect;

        inputBox.set(block.getBoundingBox());
        inputText = block.getText();
    }

    @Override
    protected void draw(@NonNull  Canvas canvas) {
        Rect rect = new Rect();

        calculateRect(inputOverlay,
                    (float)imgRect.height(),
                    (float)imgRect.width(),
                    inputBox).round(rect);

        canvas.drawText(inputText, Math.min(rect.left,rect.right), rect.top, textPaint);
        canvas.drawRect(rect, boxPaint);
    }


}
