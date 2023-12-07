package com.example.face_detector.camera;

import static java.lang.Math.ceil;
import static java.lang.Math.round;

import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.icu.number.Scale;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;

import com.google.mlkit.vision.face.Face;

import java.lang.reflect.Array;
import java.nio.file.attribute.GroupPrincipal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

public class GraphicOverlay extends View {

    private static Float mScale = null;
    private static Float mOffsetX = null;
    private static Float mOffsetY = null;
    private static int cameraSelector = CameraSelector.LENS_FACING_FRONT;
    private ArrayList<Graphic> graphics = new ArrayList<Graphic>();


    public GraphicOverlay(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }




//    public boolean isFrontMode() {
//       return  cameraSelector == CameraSelector.LENS_FACING_FRONT;
//    }

    public static boolean isFrontMode() {
        return cameraSelector == CameraSelector.LENS_FACING_FRONT;
    }

    public static abstract class Graphic {
        private GraphicOverlay overlay;
        private Context context;


        protected abstract void draw(Canvas canvas);

        public void getContext(GraphicOverlay view) {
            context = view.getContext();
        }
        private boolean isLandScapeMode() {
            return context.getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        }

        private float ifLandScapeMode(Float mode, Float other) {
            if (isLandScapeMode())
                return mode;
            else
                return other;
        }

        public RectF calculateRect(GraphicOverlay ROverlay, Float height, Float width, RectF boundingBoxT) {
            float screenWidth = ROverlay.getWidth();
            float screenHeight = ROverlay.getHeight();

            Float scaleX = screenWidth / ifLandScapeMode(width, height);
            Float scaleY = screenHeight / ifLandScapeMode(height, width);
            Float scale = scaleX < scaleY ? scaleY : scaleX;

            mScale = new Float(scale);
            Float offsetX = (screenWidth - (float)ceil(ifLandScapeMode(width, height) * scale)) / 2.0f;
            Float offsetY = (screenHeight - (float)ceil(ifLandScapeMode(height, width) * scale)) / 2.0f;
            mOffsetX = offsetX;
            mOffsetY = offsetY;

            RectF mapped = new RectF();
            mapped.set(boundingBoxT.right * scale + offsetX, boundingBoxT.top * scale + offsetY,
                    boundingBoxT.left * scale + offsetX, boundingBoxT.bottom * scale + offsetY);
            if (isFrontMode()) {
                Float centerX = screenWidth / 2;
                mapped.left = centerX + (centerX - mapped.left);
                mapped.right = centerX - (mapped.right - centerX);
            }

            return mapped;
        }
    }

    protected void toggleSelector() {
        cameraSelector = cameraSelector == CameraSelector.LENS_FACING_BACK ? CameraSelector.LENS_FACING_FRONT
                                                                           : CameraSelector.LENS_FACING_BACK;
    }



    public synchronized void clear() {
        graphics.clear();
        postInvalidate();
    }

    public synchronized void add(Graphic graphic) {
            graphics.add(graphic);
    }

    public synchronized void remove(Graphic graphic) {
            graphics.remove(graphic);
            postInvalidate();
        }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (Graphic graphic : graphics) {
            graphic.draw(canvas);
        }
    }

}
