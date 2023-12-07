package com.example.mlkit_ocr;

import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;


public class MLKITActivity extends AppCompatActivity {

    private File photoFile;
    private static final String TAG = MLKITActivity.class.getName();
    TextView Label, Time;
    private TextRecognizer recognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
    final int TAKE_PHOTO = 1;
    final int FROM_STORAGE = 2;
    final int REQUEST_CODE_PERMISSIONS = 0;
    private static final int MY_CAMERA_REQUEST_CODE = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        File directory = new File(getFilesDir(), "photos");
        if (!directory.exists()) {
            directory.mkdir();
        }

        if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 0);


        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
        }

        photoFile = null;
        try {
            photoFile = createImageFile();
        } catch (IOException ex) {
            Log.e("MLKIT", ex.getMessage(), ex);
        }
        setContentView(R.layout.activity_mlkit);
        Time = findViewById(R.id.inference_time);
        Label = findViewById(R.id.Label);
        Label.setMovementMethod(ScrollingMovementMethod.getInstance());
        Button pickPicture = findViewById(R.id.btn_pick_pic);
        Button takePicture = findViewById(R.id.btn_take_pic);
        pickPicture.setOnClickListener(v -> onPickImage());
        takePicture.setOnClickListener(v -> onTakeImage());

    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.i("REQUEST"," " + requestCode);
        switch (requestCode) {
            case REQUEST_CODE_PERMISSIONS:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
                finish();
                 }
        }
    }
    private void onTakeImage() {
        Uri uri = FileProvider.getUriForFile(this, "com.example.Mlkit_ocr.fileprovider", photoFile);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
        startActivityForResult(intent, TAKE_PHOTO);
    }
    private  void onPickImage() {
        Uri uri = FileProvider.getUriForFile(this, "com.example.Mlkit_ocr.fileprovider", photoFile);
        Intent intent = new Intent();

        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        intent = Intent.createChooser(intent,"Select files");

        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, FROM_STORAGE);
        }
    }

    public static double calculateInSampleSize(
            BitmapFactory.Options options, int TARGET_WIDTH, int TARGET_HEIGHT) {
        double inSampleSize = 0;
        boolean scaleByHeight = Math.abs(options.outHeight - TARGET_HEIGHT) >= Math.abs(options.outWidth - TARGET_WIDTH);


        inSampleSize = scaleByHeight
                ? options.outHeight / TARGET_HEIGHT
                : options.outWidth / TARGET_WIDTH;

        return inSampleSize;
    }


    @Override
    protected void onActivityResult(int RequestCode, int resultCode, Intent data) {

        super.onActivityResult(RequestCode, resultCode, data);
        {
            if (resultCode == Activity.RESULT_OK) {
                //Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                Bitmap bitmap = null;
                Bitmap InputMap = null;
                int TARGET_HEIGHT = 800 * 2;
                int TARGET_WIDTH = 405 * 2;
                Matrix matrix = new Matrix();
                int Rotation = 0;

                if (RequestCode == TAKE_PHOTO) {
                    InputMap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(photoFile.getAbsolutePath(), options);
                    double sampleSize = 0;
                    if (options.outWidth > options.outHeight) {
                        sampleSize = calculateInSampleSize(options, TARGET_HEIGHT, TARGET_WIDTH);
                        Rotation = 90;
                    }
                    else {
                        //Rotation = 90;
                        sampleSize = calculateInSampleSize(options, TARGET_WIDTH, TARGET_HEIGHT);
                    }

                    options.inSampleSize = (int)Math.pow(2d,
                            Math.floor(Math.log(sampleSize)/Math.log(2d)));

                    options.inJustDecodeBounds = false;
                    bitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath(), options);
                    matrix.postRotate(Rotation);
                    bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                } else {
                    Uri selectedImageUri = data.getData();

                    if (selectedImageUri != null) {
                        try {

                            InputStream inputStream = getContentResolver().openInputStream(selectedImageUri);

                            inputStream = getContentResolver().openInputStream(selectedImageUri);
                            BitmapFactory.Options options = new BitmapFactory.Options();
                            BitmapFactory.Options optionsMap = new BitmapFactory.Options();
                            options.inJustDecodeBounds = true;
                            optionsMap.inJustDecodeBounds = true;

                            BitmapFactory.decodeStream(inputStream, null, options);
                            BitmapFactory.decodeStream(inputStream, null, optionsMap);
                            Boolean scaleByHeight = Math.abs(options.outHeight - TARGET_HEIGHT) >= Math.abs(options.outWidth - TARGET_WIDTH);
                            double sampleSize = 0;
                            double MapSampleSize = calculateInSampleSize(optionsMap,288,288);
                            if (options.outWidth > options.outHeight) {

                                sampleSize = scaleByHeight
                                        ? options.outHeight / TARGET_WIDTH
                                        : options.outWidth / TARGET_HEIGHT;
                                Rotation = 90;
                            }
                            else {
                                sampleSize = scaleByHeight
                                        ? options.outHeight / TARGET_HEIGHT
                                        : options.outWidth / TARGET_WIDTH;


                            }

                            options.inSampleSize = (int)Math.pow(2d,
                                    Math.floor(Math.log(sampleSize)/Math.log(2d)));
                            optionsMap.inSampleSize = (int)Math.pow(2d,
                                    Math.floor(Math.log(MapSampleSize)/Math.log(2d)));
                            options.inJustDecodeBounds = false;
                            optionsMap.inJustDecodeBounds = false;
                            inputStream = getContentResolver().openInputStream(selectedImageUri);

                            bitmap = BitmapFactory.decodeStream(inputStream, null, options);
                            inputStream = getContentResolver().openInputStream(selectedImageUri);
                            InputMap = BitmapFactory.decodeStream(inputStream, null, optionsMap);
                            matrix.postRotate(Rotation);
                            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                        } catch (FileNotFoundException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    else {
                        Toast.makeText(getApplicationContext(), "pain", Toast.LENGTH_LONG).show();
                    }
                }
                if (bitmap != null) {
                    Log.i("MYTAG", "IMAGE RESOLUTION: " + bitmap.getWidth() + " x " + bitmap.getHeight());

                    ImageView ivPhoto = findViewById(R.id.iv_photo);
                    //ivPhoto.setRotation(ivPhoto.getRotation() + Rotation);
                    ivPhoto.setImageBitmap(bitmap);


                    Date start = new Date();
                    ProgressDialog MLKITRunModel = ProgressDialog.show(this, "", "running model...", false, false);
                    InputImage inputImage = InputImage.fromBitmap(InputMap, 0);


                    recognizer.process(inputImage).addOnSuccessListener(new OnSuccessListener<Text>() {
                        @Override
                        public void onSuccess(@NonNull Text visionText) {

                            // get visionText position on screen
                            Date end = new Date();
                            float preprocessTime = (float) (end.getTime() - start.getTime());
                            String inference = "Inference time: " + preprocessTime + " ms";
                            String Messeage = "";
                            for (Text.TextBlock block : visionText.getTextBlocks()) {
                                String blockText = block.getText();
                                for (Text.Line line : block.getLines()) {
                                    String lineText = line.getText();

                                    Messeage += lineText + "\n";
                                }
                            }
                            Time.setText(inference);
                            Label.setText(Messeage);


                        }
                    }).addOnCompleteListener(new OnCompleteListener<Text>() {
                        @Override
                        public void onComplete(@NonNull Task<Text> task) {
                            if(MLKITRunModel!=null && MLKITRunModel.isShowing()) {
                                MLKITRunModel.dismiss();
                            }
                        }
                    });

                    // Cleanup
                    photoFile.delete();
                }
            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".bmp",         /* suffix */
                storageDir      /* directory */
        );

        return image;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

}