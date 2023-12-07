package com.example.mlkit_ocr.textDectector;


import android.content.Context;
import android.graphics.Rect;
import android.widget.TextView;
import android.widget.Toast;

import com.example.mlkit_ocr.camera.GraphicOverlay;
import com.example.mlkit_ocr.camera.imageAnalyzer;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.util.Date;

public class viewDetector extends imageAnalyzer<Text> {

    private TextRecognizer recognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
    private Context viewContext;
    public TextView rps_tv;
    public int count = 0;
    public Date start;
    public Date end;

    public viewDetector(GraphicOverlay view, Context context, TextView tv) {
        getOverlay(view);
        viewContext = context;
        rps_tv = tv;
    }
    @Override
    protected Task<Text> detectInImage(InputImage inputImage) {
        start = new Date();

        Task<Text> result = recognizer.process(inputImage).addOnSuccessListener(
                new OnSuccessListener<Text>() {
                    @Override
                    public void onSuccess(Text Texts) {
                        Date end = new Date();
                        Float processedTime = (float)(end.getTime() - start.getTime());
                        rps_tv.setText("RPS: " + (int)(1000 / processedTime) + "");
                    }
                });

        return result;
    }

    @Override
    protected void stop() {
        recognizer.close();
    }

    @Override
    protected void onSuccess(Text result, GraphicOverlay graphicOverlay, Rect rect) {
        graphicOverlay.clear();
        try {
            for (Text.TextBlock block : result.getTextBlocks()) {
                GraphicOverlay.Graphic TextG = new viewGraphic(graphicOverlay, block, rect);
                graphicOverlay.add(TextG);
            }
            graphicOverlay.postInvalidate();
        }
        catch (Exception e) {
            Toast.makeText(viewContext, "Draw box failed! Exception:" + e, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onFailure(Exception e) {
        Toast.makeText(viewContext, "Detect image failed! Exception: " + e, Toast.LENGTH_LONG).show();
    }

}
