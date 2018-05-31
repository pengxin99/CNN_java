package CNN_JAVA;

import CNN_JAVA.Tensor;

import javax.imageio.stream.FileImageInputStream;
import javax.imageio.stream.FileImageOutputStream;
import java.io.*;
import java.util.ArrayList;
import java.awt.image.BufferedImage;

public class util {
    // activation
    static public Tensor Activation(Tensor input, String type){
        for (double[][] data : input.getTensor()) {
            int size = input.getTensor().get(0).length;
//            double val_temp = 0.0;
            if ("sigmoid".equals(type)){
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        data[i][j] = sigmoid(data[i][j]);
//                        input.setTensorByPixel(i,j,val_temp);
                    }
                }
            }else if ("tanh".equals(type)){
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        data[i][j] = tanh(data[i][j]);
//                        input.setTensorByPixel(i,j,val_temp);
                    }
                }
            }else if ("relu".equals(type)){
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        data[i][j] = relu(data[i][j]);
//                        input.setTensorByPixel(i,j,val_temp);
                    }
                }
            }
        }

        return input;
    }

    static public double sigmoid(double val){
        return 1.0/(1.0+Math.exp(-val));
    }
    static public double tanh(double val){
        return Math.tanh(val);
    }
    static public double relu(double val){
        return Math.max(0.0,val);
    }

    static public double[] softmax(double[] val){
        double sum = 0.0;
        for (double v:val ) {
            sum += Math.exp(v);
        }
        for (int i = 0; i < val.length; i++) {
            val[i] = Math.exp(val[i]) / sum ;
        }
        return val;
    }

    static public ArrayList<String> ReadTxt(String filepath){
        ArrayList<String> para = new ArrayList<>();

        try { // 防止文件建立或读取失败，用catch捕捉错误并打印，也可以throw

//            /* 写入Txt文件 */
//            File writename = new File("E:\\Java project\\CNN\\src\\output.txt"); // 相对路径，如果没有则要建立一个新的output。txt文件
//            writename.createNewFile();                                       // 创建新文件
//            BufferedWriter out = new BufferedWriter(new FileWriter(writename));


            /* 读入TXT文件 */
//            filepath = "E:\\Java project\\CNN\\src\\para.txt";  // 绝对路径或相对路径都可以，这里是绝对路径，写入文件时演示相对路径
            File filename = new File(filepath);                 // 要读取以上路径的input。txt文件
            InputStreamReader reader = new InputStreamReader(
                    new FileInputStream(filename));             // 建立一个输入流对象reader
            BufferedReader br = new BufferedReader(reader);     // 建立一个对象，它把文件内容转成计算机能读懂的语言
            String line = "";

            line = br.readLine();
            para.add(line) ;
            while (line != null) {
                line = br.readLine(); // 一次读入一行数据
//                out.write(line+'\n');
//                out.flush();
                para.add(line);
            }


//            out.write("我会写入文件啦\r\n");                               // \r\n即为换行
//            out.flush();                                                     // 把缓存区内容压入文件
//            out.close();                                                     // 最后记得关闭文件

        } catch (Exception e) {
            e.printStackTrace();
        }
        return para;
    }
    // 打印读入的卷积层1的参数
    public static void printPara(double[][] conv2d_1_1){
        for (int i = 0; i < conv2d_1_1.length; i++) {
            for (int j = 0; j < conv2d_1_1[0].length; j++) {
                System.out.print(conv2d_1_1[i][j]+" ");
            }
            System.out.println();
        }
    }
    // 打印读入的卷积层2的参数
    public static void printPara2(double[][][] conv2d_1_1){
        for (int i = 0; i < conv2d_1_1.length; i++) {
            for (int j = 0; j < conv2d_1_1[0].length; j++) {
                System.out.print(conv2d_1_1[i][j][0]+" ");
                System.out.print(conv2d_1_1[i][j][1]+" ");
                System.out.println();
            }
            System.out.println();
        }
    }

    //图片到byte数组
    public static byte[] image2byte(String path){
        byte[] data = null;
        FileImageInputStream input = null;

        try {
            input = new FileImageInputStream(new File(path));
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int numBytesRead = 0;
            while ((numBytesRead = input.read(buf)) != -1) {
                output.write(buf, 0, numBytesRead);
            }
            data = output.toByteArray();
            output.close();
            input.close();
        }
        catch (FileNotFoundException ex1) {
            ex1.printStackTrace();
        }
        catch (IOException ex1) {
            ex1.printStackTrace();
        }
        return data;
    }
    //byte数组到图片
    public static void byte2image(byte[] data,String path){
        if(data.length<3||path.equals("")) return;
        try{
            FileImageOutputStream imageOutput = new FileImageOutputStream(new File(path));
            imageOutput.write(data, 0, data.length);
            imageOutput.close();
            System.out.println("Make Picture success,Please find image in " + path);
        } catch(Exception ex) {
            System.out.println("Exception: " + ex);
            ex.printStackTrace();
        }
    }
}
