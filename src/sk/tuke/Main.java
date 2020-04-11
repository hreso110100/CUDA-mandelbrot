package sk.tuke;

import edu.rit.color.HSB;
import edu.rit.image.PJGColorImage;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import javax.imageio.ImageIO;

class Main {

    static int height = 800, width = 800;

    public static void main(String[] args) throws Exception {
        int[][] matrix = new int[height][width];
        BufferedReader reader = new BufferedReader(new FileReader("cuda/output.txt"));

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                String[] r = reader.readLine().split(",");
                matrix[i][j] = HSB.pack(Float.parseFloat(r[0]), Float.parseFloat(r[1]), Float.parseFloat(r[2]));
            }
        }
        BufferedImage image = new PJGColorImage(height, width, matrix).getBufferedImage();

        File outputFile = new File("result.png");
        ImageIO.write(image, "png", outputFile);

    }
}