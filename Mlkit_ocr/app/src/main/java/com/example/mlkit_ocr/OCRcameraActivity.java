package com.example.mlkit_ocr;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;

import com.codemonkeylabs.fpslibrary.TinyDancer;
import com.example.mlkit_ocr.camera.CameraManager;
import com.example.mlkit_ocr.camera.GraphicOverlay;

public class OCRcameraActivity extends AppCompatActivity {

    private CameraManager cameraManager;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mlkit_camera);
        PreviewView cameraPreview = findViewById(R.id.previewView_finder);
        GraphicOverlay cameraOverlay = findViewById(R.id.graphicOverlay_finder);
        Button switchCamera = findViewById(R.id.btn_switch_camera);
        cameraPreview.setScaleX(-1F);
        TinyDancer.create().show(this);

        TinyDancer.create()
                .redFlagPercentage(.1f) // set red indicator for 10%
                .startingGravity(Gravity.TOP)
                .startingXPosition(100)
                .startingYPosition(10)
                .show(this);

        TextView tv = findViewById(R.id.aps);
        checkCameraPermission();
        cameraManager = new CameraManager(this, this, cameraPreview, cameraOverlay, tv);
        switchCamera.setOnClickListener(v -> cameraManager.setCameraLen());
    }


    private void checkCameraPermission() {
        if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 0);
    }
}