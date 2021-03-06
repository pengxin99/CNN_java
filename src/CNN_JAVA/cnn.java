package CNN_JAVA;

import CNN_JAVA.util.*;
import sun.rmi.server.Activation;

import javax.imageio.ImageIO;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;

interface Layer {
    enum LayerType {
        // 网络层的类型：输入层、输出层、卷积层、采样层
        input, output, conv, samp
    }

    void run_forward();

    void print_output();

    Tensor getOutputTensor();
}

class Tensor {
    // 因为每次层与层之间传递的图像数据可能有多个，例如经过卷积层之后可能产生两个图片处理数据，这里使用Arraylist存放
    private ArrayList<double[][]> tensorList = new ArrayList<>();
    private double[][] tensor;

    Tensor() {
    }

    Tensor(double[][] data) {
        this.tensorList.add(data);
    }

    Tensor(ArrayList<double[][]> tensorList) {
        this.tensorList = tensorList;
    }

    public void setTensor(ArrayList<double[][]> tensorList) {
        this.tensorList = tensorList;
    }

    Tensor(int height, int width) {
        this.tensor = new double[height][width];
        this.tensorList.add(this.tensor);
    }

    public ArrayList<double[][]> getTensor() {
        return this.tensorList;
    }

    public void setTensorByPixel(int index, int i, int j, double val) {
        this.getTensor().get(index)[i][j] = val;
    }

    public void printTensor() {
        for (double[][] tensor : tensorList) {
            for (int i = 0; i < tensor.length; i++) {
                for (int j = 0; j < tensor[0].length; j++) {
                    System.out.print(tensor[i][j] + "\t");
                }
                System.out.println();
            }
            System.out.println("********************************");
        }

    }
}

class CONV implements Layer {
    enum CONV_TYPE {same, vaild, full}

    private LayerType Conv;
    private String TYPE;
    private int FilterNum;                  // 卷积层中有几个卷积核
    private int FilterSize;
    private ArrayList<double[][][]> FilterPara;
    private ArrayList<double[]> Bias;      // 每个卷积滤波器设置一个bias
    private Tensor inputTensor;
    private Tensor outputTensor = new Tensor();

    public CONV(Tensor inputTensor, int filternum, int filtersize, String type) {
        this.inputTensor = inputTensor;
        this.FilterNum = filternum;
        this.FilterSize = filtersize;
        this.TYPE = type;
    }

    public void setParameters(ArrayList<double[][][]> para, ArrayList<double[]> bias) {
        this.FilterPara = para;
        this.Bias = bias;
    }


    public void printPara() {
        for (double[][][] p : FilterPara) {
            for (int i = 0; i < p.length; i++) {
                for (int j = 0; j < p[0].length; j++) {
                    for (int k = 0; k < p[0][0].length; k++) {
                        System.out.println(p[i][j][k]);
                    }
                }
            }
        }
    }

    public Tensor getOutputTensor() {
        return this.outputTensor;
    }

    // 更具 type 类型相应改变输入数据类型
    public void SAME() {
        ArrayList<double[][]> input_same = new ArrayList<>();
        int InputSize = this.inputTensor.getTensor().get(0).length;
        int AddSize = this.FilterSize - 1;
        for (double[][] InitInput : this.inputTensor.getTensor()) {

            double[][] newInput = new double[InputSize + AddSize][InputSize + AddSize];
            for (int i = 0; i < newInput.length; i++) {
                for (int j = 0; j < newInput[0].length; j++) {
                    // 如果是不需要扩展的部分，直接将原始数组拿来即可
                    if (i >= AddSize / 2 && j >= AddSize / 2 && i < InputSize + AddSize / 2 && j < InputSize + AddSize / 2) {
                        newInput[i][j] = InitInput[i - AddSize / 2][j - AddSize / 2];
                    } else {
                        // 因为same需要扩展的部分，补0
                        newInput[i][j] = 0;
                    }
                }
            }
            input_same.add(newInput);
        }
        this.inputTensor.setTensor(input_same);
    }

    @Override
    public void run_forward() {
        if (this.TYPE.equals("same")) {
            SAME();
        }
        int TensorNum = this.inputTensor.getTensor().size();
        int FilterNum = this.FilterPara.size();

        double[][][] tensor = new double[TensorNum][][];
        double[][][][] filter = new double[FilterNum][this.FilterPara.get(0)[0][0].length][this.FilterSize][this.FilterSize];
        // 将数据和卷积核从 ArrayList 中提取出来，放入数组
        for (int i = 0; i < TensorNum; i++) {
            tensor[i] = this.inputTensor.getTensor().get(i);
        }
        for (int i = 0; i < FilterNum; i++) {
            for (int j = 0; j < this.FilterSize; j++) {
                for (int k = 0; k < this.FilterSize; k++) {
                    for (int l = 0; l < this.FilterPara.get(0)[0][0].length; l++) {
                        // 将 5*5*1改为1*5*5
                        filter[i][l][j][k] = this.FilterPara.get(i)[j][k][l];
                    }
                }
            }
        }
        // 每个卷积核对输入图像进行卷积操作，得到 filternum * tensorlistnum 个结果

        double[][][] OutPutList = new double[FilterNum][][];
        for (int i = 0; i < FilterNum; i++) {
            double[][][] temp = new double[FilterNum * TensorNum][][];
            int index = 0;
            for (int j = 0; j < TensorNum; j++) {
//                this.outputTensor.getTensor().add(convRes(tensor[j],filter[i][j], this.Bias.get(i)[0]));
                temp[index] = convRes(tensor[j], filter[i][j], this.Bias.get(i)[0]);
                index++;
            }
            double[][] data = new double[temp[0].length][temp[0].length];
            for (int j = 0; j < data.length; j++) {
                for (int k = 0; k < data.length; k++) {
                    for (int l = 0; l < index; l++) {
                        data[j][k] += temp[l][j][k];
                    }
                }
            }
            OutPutList[i] = data;
        }
        // 将所有卷积输出结果add到outputTensor，数量应该是卷积核的个数
        for (double[][] data :
                OutPutList) {
            outputTensor.getTensor().add(data);
        }

    }

    // 计算单个输入和单个卷积核的卷积结果
    public double[][] convRes(double[][] InputPic, double[][] filter, double bias) {
        int InputWidth = InputPic.length;
        int InputHeight = InputPic[0].length;
        int OutputWidth = InputWidth - filter.length + 1;
        int OutputHeight = InputHeight - filter.length + 1;
        // 设定输出向量的临时值
        double[][] output_temp = new double[OutputHeight][OutputWidth];

        // filter在输入tensor上滑动
        for (int j = 0; j < OutputWidth; j++) {
            for (int i = 0; i < OutputHeight; i++) {
                // 每次滑动计算生成的单个神经元的值
                double temp = 0.0;
                for (int k = 0; k < this.FilterSize; k++) {
                    for (int l = 0; l < this.FilterSize; l++) {
                        temp += InputPic[i + k][j + l] * filter[k][l];
                    }
                }
                // 记录单次滑动卷积层生成的值
                output_temp[i][j] = temp + bias;
            }
        }
        return output_temp;
    }

    @Override
    public void print_output() {
        System.out.println("**************  After the Conv layer  ************************");
        outputTensor.printTensor();
    }
}

class POOL implements Layer {
    enum POOL_TYPE {max, average}

    private POOL_TYPE type;
    private int PoolSize;
    private Tensor InputTensor;
    private Tensor OutputTensor = new Tensor();

    public POOL() {
    }

    public POOL(Tensor input, POOL_TYPE type, int poolSize) {
        this.InputTensor = input;
        this.type = type;
        this.PoolSize = poolSize;
    }

    public void padding() {
        // 设置新的临时tensor，用于保存padding过后的数据
        Tensor temp_tensor = new Tensor();
        int size = this.InputTensor.getTensor().get(0)[0].length;
        for (int i = 0; i < this.InputTensor.getTensor().size(); i++) {
            temp_tensor = new Tensor(this.InputTensor.getTensor());

            // 如果输入tensor的shape不是偶数，则进行增加一行，一列的padding操作
            if (size / 2 != 0) {

                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        temp_tensor.setTensorByPixel(i, j, k, this.InputTensor.getTensor().get(i)[j][k]);
                    }
                }
                // padding新加的一行和一列
                for (int l = 0; l < size + 1; l++) {
                    temp_tensor.setTensorByPixel(i, l, size, 0);
                    temp_tensor.setTensorByPixel(i, size, l, 0);
                }
                this.InputTensor = temp_tensor;
            }
        }
    }

    @Override
    public void run_forward() {
        // 在前向传播之前先进行padding
//        padding();
        int inputsize = InputTensor.getTensor().get(0).length;
        int outputsize = inputsize / this.PoolSize;
        ArrayList<double[][]> output_ArrayList = new ArrayList<>();

        for (double[][] data :
                this.InputTensor.getTensor()) {
            double[][] output_temp = new double[outputsize][outputsize];
            int x = 0;
            int y = 0;
            for (int i = 0; i < inputsize; i += this.PoolSize) {
                // 如果这次循环结束时越界，则舍弃
                if (i + this.PoolSize - 1 >= inputsize)
                    break;
                for (int j = 0; j < inputsize; j += this.PoolSize) {
                    // 如果这次循环结束时越界，则舍弃
                    if (j + this.PoolSize - 1 >= inputsize)
                        break;
                    double max = findMax(data, i, j);
                    output_temp[x][y] = max;
                    // 一次输入到outputTensor中，以x为索引
                    x++;
                }
                // 开始新一的行，x置0，逐行扫描
                x = 0;
                y++;
            }
            output_ArrayList.add(output_temp);
        }
        OutputTensor.setTensor(output_ArrayList);
    }

    @Override
    public void print_output() {
        System.out.println("***********************  After the Pooling layer  **********************");
        OutputTensor.printTensor();
    }

    public double findMax(double[][] data, int x, int y) {
        double temp_max = -Double.MAX_VALUE;
        for (int i = x; i < x + this.PoolSize; i++) {
            for (int j = y; j < y + this.PoolSize; j++) {
                if (data[i][j] > temp_max) {
                    temp_max = data[i][j];
                }
            }
        }
        return temp_max;
    }

    public Tensor getOutputTensor() {
        return this.OutputTensor;
    }
}

class FLATTEN implements Layer {
    private Tensor InputTensor;
    private Tensor OutputTensor;
    private int InputNum;
    private int InputSize;
    private int OutputSize;
    private int index = 0;

    public FLATTEN() {
    }

    public FLATTEN(Tensor input) {
        this.InputTensor = input;
    }

    public Tensor getOutputTensor() {
        return this.OutputTensor;
    }

    @Override
    public void run_forward() {
        this.InputNum = InputTensor.getTensor().size();
        this.InputSize = InputTensor.getTensor().get(0).length;
        this.OutputSize = this.InputNum * this.InputSize * this.InputSize;
        OutputTensor = new Tensor(this.OutputSize, 1);

        for (double[][] data :
                this.InputTensor.getTensor()) {
            for (int i = 0; i < this.InputSize; i++) {
                for (int j = 0; j < this.InputSize; j++) {
                    OutputTensor.setTensorByPixel(0, index, 0, data[i][j]);
                    index++;
                }
            }
        }
    }

    @Override
    public void print_output() {
        System.out.println("***************** After the Flatten layer *** the length of the output is: " + index);
        OutputTensor.printTensor();
    }
}

class DENSE implements Layer {
    private Tensor InputTensor;
    private Tensor OutputTensor;
    private double[][] DensePara;
    private double[] DenseBias;

    public DENSE() {
    }

    public DENSE(Tensor input, int outsize) {
        this.InputTensor = input;
        this.OutputTensor = new Tensor(outsize, 1);
    }

    public void setDensePara(double[][] densePara, double[] denseBias) {
        this.DensePara = densePara;
        this.DenseBias = denseBias;
    }

    public Tensor getOutputTensor() {
        return this.OutputTensor;
    }

    @Override
    public void run_forward() {
        ArrayList<double[][]> OutputList = new ArrayList<>();
        double[][] InputData = this.InputTensor.getTensor().get(0);
        double[][] OutputData = this.OutputTensor.getTensor().get(0);
        for (int i = 0; i < OutputData.length; i++) {
            double temp = 0.0;
            for (int j = 0; j < InputData.length; j++) {
                temp += InputData[j][0] * this.DensePara[j][i];
            }
            OutputData[i][0] = temp + this.DenseBias[i];
        }
        OutputList.add(OutputData);
        this.OutputTensor.setTensor(OutputList);
    }

    @Override
    public void print_output() {
        System.out.println("****** After the Dense layer, the output size is :" + OutputTensor.getTensor().get(0).length);
        this.OutputTensor.printTensor();
    }
}


public class cnn {
    public static void main(String[] args) {

        String packageName = cnn.class.getPackage().getName();
        System.out.println(packageName);

        double[][] imageData = {{166, 166, 167, 167, 166, 167, 168, 170, 170, 168, 167, 164, 166, 142, 125, 145, 166, 141, 114, 148, 165, 144, 114, 143, 169, 148, 115, 144, 165, 163, 165, 163, 168, 151, 142, 169, 164, 164, 165, 167, 165, 168, 167, 165, 166, 167, 166, 165, 165, 171, 167, 171, 169, 173, 169, 168, 171, 171, 172, 171, 172, 172, 174, 176},
                {166, 166, 167, 167, 166, 168, 168, 169, 168, 167, 165, 163, 166, 144, 130, 149, 159, 147, 128, 154, 164, 150, 126, 153, 164, 152, 126, 150, 164, 161, 165, 166, 170, 155, 139, 169, 165, 165, 167, 167, 166, 168, 166, 166, 167, 167, 166, 166, 168, 169, 169, 170, 170, 170, 168, 172, 171, 171, 172, 171, 172, 172, 174, 175},
                {167, 167, 167, 167, 167, 168, 167, 168, 169, 168, 166, 165, 167, 145, 134, 153, 166, 144, 129, 161, 169, 151, 127, 156, 164, 153, 127, 149, 166, 164, 166, 166, 169, 161, 134, 168, 166, 166, 168, 167, 169, 169, 167, 167, 169, 168, 167, 169, 171, 169, 169, 169, 172, 171, 169, 170, 172, 172, 172, 172, 172, 172, 175, 176},
                {165, 164, 164, 164, 164, 164, 164, 165, 168, 164, 165, 165, 164, 137, 128, 150, 165, 128, 114, 156, 165, 139, 114, 144, 163, 147, 118, 139, 164, 165, 165, 163, 167, 166, 132, 164, 167, 164, 168, 166, 169, 168, 166, 168, 169, 167, 168, 173, 169, 175, 173, 174, 173, 170, 170, 170, 171, 171, 171, 171, 171, 172, 174, 175},
                {165, 166, 165, 164, 163, 164, 165, 167, 169, 164, 165, 166, 160, 128, 124, 152, 162, 126, 119, 158, 166, 141, 118, 146, 164, 147, 118, 138, 164, 164, 165, 166, 164, 168, 132, 156, 164, 159, 162, 160, 163, 162, 162, 163, 164, 162, 165, 171, 205, 196, 169, 169, 167, 167, 171, 169, 168, 169, 169, 169, 169, 170, 171, 171},
                {167, 168, 168, 167, 167, 168, 168, 171, 172, 166, 165, 166, 157, 125, 128, 159, 159, 124, 122, 157, 166, 141, 119, 146, 167, 149, 121, 138, 165, 165, 165, 167, 168, 173, 143, 154, 172, 165, 168, 167, 168, 168, 168, 169, 170, 169, 171, 178, 236, 213, 173, 174, 171, 169, 168, 166, 169, 170, 170, 170, 171, 171, 172, 173},
                {168, 167, 167, 167, 168, 169, 168, 169, 169, 164, 163, 162, 152, 124, 130, 161, 158, 123, 123, 157, 167, 141, 117, 146, 166, 148, 119, 133, 163, 165, 164, 165, 166, 169, 146, 143, 172, 165, 167, 168, 168, 168, 168, 169, 170, 169, 171, 175, 180, 176, 169, 174, 172, 172, 171, 176, 171, 172, 172, 172, 173, 174, 175, 175},
                {171, 170, 169, 169, 171, 171, 170, 168, 167, 166, 165, 162, 152, 126, 133, 161, 154, 126, 128, 157, 166, 142, 120, 147, 165, 150, 122, 131, 161, 166, 168, 170, 170, 171, 151, 138, 173, 167, 167, 167, 168, 168, 168, 169, 171, 171, 171, 173, 173, 173, 177, 170, 170, 174, 168, 170, 171, 171, 172, 172, 173, 174, 175, 175},
                {169, 170, 170, 171, 172, 172, 170, 169, 168, 163, 159, 159, 154, 137, 147, 161, 157, 137, 143, 157, 158, 148, 133, 150, 159, 156, 130, 141, 158, 161, 163, 160, 163, 164, 154, 126, 168, 167, 170, 169, 168, 169, 169, 170, 170, 170, 171, 172, 173, 174, 173, 172, 171, 171, 172, 173, 172, 173, 173, 173, 174, 175, 176, 176},
                {169, 170, 170, 171, 172, 172, 171, 170, 170, 166, 164, 164, 147, 123, 139, 163, 158, 121, 129, 162, 165, 137, 118, 147, 164, 154, 124, 136, 161, 165, 165, 164, 167, 167, 160, 127, 165, 167, 169, 170, 168, 169, 169, 170, 170, 171, 171, 171, 172, 173, 172, 172, 171, 170, 171, 172, 173, 173, 173, 174, 175, 176, 176, 177},
                {171, 171, 171, 171, 172, 173, 172, 171, 171, 163, 161, 162, 143, 123, 141, 162, 153, 119, 129, 162, 165, 136, 116, 144, 169, 152, 118, 127, 160, 165, 166, 168, 167, 169, 168, 133, 163, 170, 167, 169, 169, 169, 170, 171, 172, 172, 172, 171, 172, 172, 172, 172, 171, 171, 172, 173, 174, 174, 175, 176, 177, 178, 177, 178},
                {170, 170, 168, 170, 170, 170, 170, 169, 168, 166, 167, 164, 138, 123, 145, 163, 152, 120, 130, 162, 163, 135, 117, 146, 167, 153, 120, 125, 160, 164, 165, 168, 167, 169, 173, 138, 157, 173, 167, 169, 169, 169, 169, 170, 171, 171, 171, 171, 171, 172, 172, 172, 172, 171, 173, 173, 174, 174, 174, 175, 177, 178, 177, 177},
                {171, 170, 169, 170, 170, 170, 170, 169, 167, 167, 167, 163, 136, 126, 148, 166, 150, 121, 132, 162, 164, 138, 119, 145, 165, 155, 126, 127, 160, 167, 166, 167, 168, 168, 172, 140, 144, 173, 167, 170, 170, 170, 171, 171, 172, 173, 173, 172, 173, 173, 174, 174, 175, 175, 175, 175, 177, 177, 177, 177, 178, 179, 180, 180},
                {166, 166, 165, 167, 166, 167, 167, 166, 168, 165, 162, 160, 136, 128, 149, 167, 146, 119, 132, 163, 166, 140, 118, 139, 167, 156, 125, 124, 154, 167, 168, 168, 169, 168, 173, 147, 133, 172, 166, 167, 166, 165, 166, 166, 167, 167, 167, 167, 167, 167, 168, 168, 169, 170, 171, 171, 172, 172, 172, 172, 173, 174, 174, 175},
                {172, 172, 171, 173, 172, 173, 172, 171, 167, 170, 167, 163, 136, 129, 150, 166, 147, 118, 132, 166, 167, 137, 117, 143, 169, 158, 127, 126, 148, 164, 167, 169, 168, 168, 173, 156, 126, 171, 168, 168, 171, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 174, 175, 176, 177, 178, 178, 178, 178, 178, 178, 179, 179},
                {172, 172, 172, 172, 172, 172, 172, 171, 167, 171, 166, 162, 144, 145, 159, 164, 147, 133, 148, 166, 162, 146, 131, 150, 167, 160, 136, 137, 152, 165, 164, 166, 168, 167, 171, 159, 119, 170, 171, 172, 173, 173, 174, 174, 174, 173, 173, 173, 174, 174, 174, 174, 176, 178, 179, 180, 180, 180, 180, 180, 180, 180, 181, 182},
                {173, 172, 171, 172, 170, 170, 172, 172, 168, 168, 169, 161, 140, 148, 165, 167, 159, 143, 154, 165, 166, 156, 142, 160, 165, 168, 149, 142, 160, 165, 163, 165, 171, 170, 170, 171, 120, 166, 171, 171, 173, 173, 172, 172, 173, 173, 172, 172, 174, 174, 175, 176, 175, 176, 177, 179, 179, 179, 180, 180, 180, 180, 181, 183},
                {173, 172, 171, 171, 172, 172, 171, 171, 169, 169, 167, 146, 122, 136, 159, 162, 144, 122, 138, 163, 164, 138, 117, 141, 166, 159, 132, 124, 149, 164, 164, 164, 170, 170, 170, 173, 123, 159, 174, 171, 172, 172, 171, 171, 172, 173, 172, 172, 173, 173, 174, 176, 176, 177, 177, 178, 180, 180, 180, 181, 181, 181, 181, 182},
                {173, 172, 172, 172, 173, 172, 170, 170, 170, 172, 168, 141, 123, 139, 162, 161, 143, 123, 141, 166, 168, 141, 119, 143, 168, 160, 131, 123, 146, 164, 164, 167, 169, 170, 170, 174, 128, 146, 177, 170, 172, 172, 172, 171, 172, 172, 172, 173, 174, 174, 174, 176, 177, 177, 178, 178, 179, 179, 181, 182, 182, 183, 182, 182},
                {173, 172, 172, 172, 172, 173, 170, 169, 166, 169, 168, 139, 127, 145, 167, 161, 138, 124, 143, 161, 161, 139, 122, 142, 166, 162, 137, 126, 143, 161, 163, 166, 170, 170, 170, 175, 137, 134, 179, 170, 172, 173, 173, 172, 172, 173, 173, 173, 174, 175, 175, 177, 177, 177, 178, 179, 180, 180, 181, 182, 183, 184, 184, 183},
                {172, 172, 173, 171, 171, 172, 170, 169, 167, 169, 166, 135, 124, 144, 166, 163, 137, 122, 145, 168, 166, 138, 118, 142, 167, 161, 135, 124, 141, 165, 164, 166, 171, 170, 170, 175, 149, 128, 178, 171, 171, 172, 173, 173, 173, 173, 173, 173, 173, 175, 176, 177, 177, 177, 178, 180, 181, 181, 181, 180, 182, 183, 185, 184},
                {172, 172, 173, 171, 171, 173, 171, 169, 172, 170, 161, 131, 124, 148, 167, 164, 133, 121, 146, 170, 166, 137, 118, 143, 168, 163, 138, 127, 140, 167, 166, 167, 169, 169, 170, 173, 159, 123, 172, 172, 170, 171, 173, 173, 173, 173, 173, 173, 173, 175, 177, 178, 177, 177, 179, 181, 181, 182, 182, 181, 181, 183, 185, 185},
                {172, 173, 173, 170, 172, 173, 171, 169, 171, 166, 154, 127, 124, 157, 170, 162, 135, 124, 147, 168, 164, 138, 121, 145, 167, 165, 141, 128, 137, 164, 164, 165, 166, 166, 170, 171, 165, 119, 161, 171, 170, 171, 173, 173, 173, 174, 174, 175, 175, 175, 176, 179, 180, 180, 181, 182, 184, 185, 185, 183, 183, 185, 186, 185},
                {172, 173, 172, 171, 172, 174, 171, 169, 170, 168, 155, 128, 123, 160, 172, 166, 140, 126, 147, 169, 165, 138, 120, 146, 170, 166, 141, 125, 133, 164, 165, 168, 165, 167, 171, 169, 168, 117, 154, 170, 170, 171, 173, 173, 173, 174, 175, 177, 175, 176, 176, 180, 183, 183, 183, 183, 186, 188, 188, 186, 185, 186, 187, 186},
                {173, 172, 169, 170, 170, 172, 173, 166, 171, 170, 160, 129, 134, 172, 183, 174, 142, 144, 155, 171, 168, 140, 138, 151, 167, 164, 147, 137, 143, 163, 162, 164, 164, 163, 171, 166, 172, 126, 142, 172, 169, 168, 173, 172, 171, 173, 173, 174, 174, 172, 172, 177, 182, 183, 195, 222, 218, 187, 186, 181, 186, 184, 185, 187},
                {166, 166, 164, 165, 165, 166, 165, 157, 164, 166, 154, 138, 143, 176, 180, 163, 144, 148, 165, 161, 157, 148, 140, 149, 148, 155, 151, 141, 142, 153, 149, 149, 153, 151, 151, 147, 158, 127, 137, 167, 151, 145, 154, 156, 150, 156, 161, 156, 155, 156, 160, 158, 152, 155, 192, 245, 225, 180, 168, 163, 153, 150, 151, 154},
                {173, 174, 174, 174, 173, 173, 171, 163, 155, 154, 149, 128, 134, 146, 167, 160, 126, 129, 152, 152, 152, 143, 124, 130, 138, 143, 144, 131, 136, 146, 144, 141, 146, 147, 145, 145, 156, 138, 130, 158, 147, 150, 154, 145, 140, 149, 158, 161, 157, 158, 158, 159, 163, 163, 167, 182, 177, 168, 168, 158, 142, 140, 145, 161},
                {173, 174, 173, 173, 172, 171, 170, 161, 157, 155, 152, 130, 131, 128, 148, 149, 124, 123, 141, 150, 152, 137, 123, 130, 146, 143, 141, 123, 132, 145, 148, 145, 143, 149, 148, 154, 159, 145, 120, 151, 140, 151, 155, 145, 143, 151, 154, 156, 153, 157, 156, 151, 156, 164, 164, 166, 162, 164, 163, 156, 158, 164, 159, 163},
                {175, 173, 172, 172, 171, 171, 171, 162, 154, 154, 141, 131, 135, 150, 149, 149, 128, 129, 148, 150, 149, 137, 125, 134, 146, 144, 147, 126, 132, 145, 153, 154, 148, 151, 151, 157, 152, 145, 115, 153, 153, 150, 155, 155, 153, 159, 160, 154, 156, 156, 157, 158, 163, 166, 164, 168, 161, 157, 166, 170, 171, 172, 172, 173},
                {173, 172, 169, 170, 170, 171, 169, 159, 153, 156, 135, 121, 132, 154, 146, 146, 123, 125, 154, 151, 146, 136, 116, 132, 142, 142, 147, 127, 130, 143, 148, 150, 153, 155, 153, 155, 148, 147, 117, 152, 160, 150, 150, 153, 156, 155, 153, 156, 151, 153, 158, 161, 161, 158, 155, 160, 164, 157, 160, 165, 170, 170, 169, 170},
                {171, 170, 168, 170, 170, 169, 165, 154, 143, 153, 134, 120, 136, 146, 144, 148, 125, 124, 150, 153, 148, 134, 118, 134, 148, 144, 146, 125, 126, 139, 139, 138, 148, 152, 152, 151, 149, 154, 119, 138, 152, 151, 149, 148, 154, 152, 147, 155, 156, 156, 161, 162, 164, 164, 157, 153, 153, 151, 142, 141, 152, 165, 166, 170},
                {171, 170, 169, 172, 171, 169, 162, 148, 150, 158, 126, 126, 138, 144, 138, 148, 123, 125, 142, 149, 150, 133, 121, 124, 146, 144, 147, 128, 127, 143, 143, 142, 144, 150, 151, 147, 153, 163, 125, 128, 154, 148, 144, 139, 143, 155, 158, 156, 150, 154, 162, 161, 161, 163, 162, 161, 150, 140, 132, 140, 145, 167, 167, 158},
                {172, 173, 172, 169, 171, 167, 158, 152, 155, 152, 127, 126, 136, 141, 135, 145, 120, 125, 141, 146, 151, 136, 121, 128, 142, 147, 146, 129, 125, 141, 146, 144, 150, 152, 148, 146, 150, 158, 138, 125, 158, 151, 144, 140, 145, 153, 159, 156, 155, 147, 151, 158, 158, 157, 157, 158, 163, 154, 149, 162, 160, 168, 164, 159},
                {171, 171, 170, 168, 168, 166, 158, 151, 155, 146, 121, 128, 139, 143, 141, 146, 119, 130, 141, 147, 152, 136, 114, 126, 143, 141, 148, 128, 126, 135, 142, 146, 147, 142, 149, 152, 151, 155, 137, 116, 155, 157, 157, 159, 158, 155, 157, 157, 160, 156, 155, 156, 160, 161, 159, 162, 162, 166, 161, 166, 174, 172, 172, 170},
                {173, 173, 171, 170, 170, 166, 158, 152, 152, 146, 125, 135, 141, 141, 142, 143, 119, 131, 140, 144, 145, 136, 115, 124, 141, 140, 145, 122, 124, 136, 143, 144, 136, 107, 129, 154, 150, 153, 149, 118, 155, 158, 155, 157, 157, 155, 155, 156, 159, 162, 161, 157, 162, 164, 162, 166, 160, 166, 160, 157, 169, 170, 166, 164},
                {172, 171, 170, 170, 168, 162, 154, 149, 159, 150, 124, 131, 139, 141, 147, 147, 125, 133, 146, 149, 144, 142, 125, 124, 129, 135, 136, 130, 130, 143, 144, 139, 146, 133, 141, 149, 153, 154, 150, 110, 145, 155, 150, 150, 153, 156, 156, 154, 152, 158, 162, 160, 161, 163, 162, 165, 168, 162, 160, 165, 166, 169, 160, 161},
                {171, 171, 169, 170, 169, 160, 152, 147, 145, 149, 133, 140, 145, 140, 148, 150, 139, 140, 153, 149, 148, 142, 133, 135, 140, 139, 131, 132, 127, 135, 142, 145, 146, 145, 149, 152, 151, 149, 157, 115, 132, 153, 150, 149, 148, 151, 155, 156, 148, 149, 157, 160, 159, 161, 161, 160, 161, 160, 154, 161, 165, 168, 173, 175},
                {171, 171, 171, 172, 169, 160, 152, 149, 150, 148, 131, 131, 132, 134, 146, 149, 131, 136, 147, 147, 155, 138, 123, 135, 141, 144, 143, 135, 129, 133, 145, 148, 140, 148, 141, 147, 156, 142, 144, 121, 126, 153, 152, 153, 151, 151, 153, 158, 155, 149, 154, 159, 160, 164, 164, 160, 163, 167, 164, 191, 207, 175, 175, 174},
                {169, 170, 169, 169, 166, 156, 150, 149, 159, 136, 120, 128, 136, 143, 144, 130, 110, 124, 141, 148, 156, 138, 112, 121, 134, 142, 145, 126, 119, 137, 150, 155, 143, 145, 144, 147, 148, 147, 153, 124, 121, 154, 153, 157, 158, 157, 153, 151, 158, 153, 153, 155, 160, 164, 165, 164, 154, 149, 161, 209, 226, 178, 167, 170},
                {172, 171, 171, 170, 166, 156, 150, 150, 158, 127, 120, 135, 140, 147, 148, 129, 113, 125, 139, 142, 140, 134, 116, 121, 130, 136, 143, 128, 114, 136, 133, 143, 144, 138, 142, 146, 143, 146, 154, 136, 116, 152, 146, 143, 145, 153, 151, 148, 153, 153, 151, 148, 154, 160, 159, 164, 162, 153, 154, 157, 156, 161, 169, 169},
                {163, 165, 165, 162, 161, 152, 150, 149, 157, 128, 125, 134, 142, 141, 146, 126, 112, 124, 138, 135, 142, 134, 111, 122, 132, 131, 145, 132, 112, 136, 142, 143, 141, 144, 139, 136, 145, 151, 146, 139, 106, 157, 151, 147, 148, 150, 154, 156, 155, 156, 157, 161, 164, 166, 167, 168, 169, 169, 171, 173, 176, 177, 178, 179},
                {175, 174, 174, 174, 170, 152, 145, 144, 158, 127, 127, 130, 139, 144, 148, 125, 116, 129, 141, 136, 141, 136, 114, 122, 132, 125, 145, 132, 114, 126, 139, 139, 137, 142, 143, 143, 147, 146, 141, 139, 114, 165, 170, 166, 169, 169, 173, 174, 172, 173, 175, 179, 180, 180, 180, 180, 182, 182, 183, 185, 186, 186, 187, 187},
                {194, 191, 189, 191, 182, 157, 147, 148, 158, 127, 128, 134, 142, 144, 146, 126, 114, 126, 141, 137, 142, 136, 112, 123, 133, 140, 149, 129, 113, 127, 140, 146, 144, 145, 143, 142, 143, 141, 141, 144, 106, 151, 171, 164, 167, 165, 169, 171, 171, 171, 171, 172, 174, 175, 176, 177, 179, 180, 181, 182, 182, 182, 182, 182},
                {192, 188, 187, 190, 179, 155, 147, 152, 155, 121, 127, 134, 142, 138, 140, 124, 114, 124, 140, 144, 149, 137, 111, 124, 137, 147, 144, 134, 114, 132, 141, 145, 147, 151, 149, 145, 146, 148, 151, 157, 111, 141, 173, 163, 167, 165, 169, 170, 171, 170, 170, 171, 173, 175, 178, 180, 180, 181, 182, 184, 185, 184, 185, 185},
                {192, 191, 189, 191, 176, 154, 149, 157, 153, 119, 128, 132, 136, 133, 142, 125, 116, 124, 139, 143, 149, 135, 109, 122, 137, 129, 132, 144, 115, 129, 136, 130, 127, 140, 146, 143, 142, 145, 148, 151, 119, 128, 171, 162, 167, 166, 170, 171, 167, 168, 170, 173, 175, 176, 177, 178, 178, 179, 180, 182, 182, 183, 183, 184},
                {192, 193, 191, 191, 173, 151, 147, 158, 150, 123, 130, 126, 130, 130, 148, 125, 115, 124, 136, 134, 141, 134, 110, 121, 131, 126, 135, 144, 118, 125, 132, 128, 130, 139, 143, 140, 141, 145, 147, 149, 129, 116, 169, 164, 167, 167, 169, 169, 169, 169, 170, 172, 173, 175, 176, 178, 179, 180, 181, 181, 181, 181, 182, 184},
                {189, 191, 190, 192, 172, 150, 149, 165, 142, 123, 128, 127, 131, 129, 150, 119, 116, 125, 137, 133, 142, 137, 112, 122, 131, 139, 142, 142, 117, 127, 130, 136, 140, 139, 138, 140, 145, 147, 145, 148, 137, 106, 165, 165, 164, 165, 165, 165, 170, 169, 168, 168, 169, 171, 175, 178, 180, 180, 180, 181, 181, 181, 183, 185},
                {191, 191, 189, 189, 167, 143, 143, 163, 139, 123, 131, 135, 140, 133, 151, 118, 115, 123, 136, 137, 145, 135, 108, 119, 138, 139, 138, 146, 114, 130, 127, 135, 132, 133, 136, 146, 150, 143, 140, 145, 142, 100, 162, 166, 163, 166, 166, 167, 168, 168, 169, 170, 172, 173, 175, 178, 178, 178, 179, 179, 179, 180, 183, 185},
                {190, 190, 189, 191, 155, 144, 143, 162, 131, 124, 136, 138, 139, 137, 150, 120, 114, 126, 137, 137, 148, 134, 110, 120, 135, 142, 132, 150, 117, 132, 133, 134, 127, 135, 143, 148, 156, 161, 155, 146, 149, 105, 149, 168, 161, 166, 161, 165, 168, 170, 170, 170, 171, 173, 177, 178, 178, 180, 180, 180, 182, 182, 183, 185},
                {192, 191, 190, 186, 154, 148, 150, 165, 135, 123, 130, 140, 145, 144, 150, 119, 120, 127, 133, 135, 143, 137, 117, 123, 136, 138, 126, 134, 124, 128, 131, 132, 132, 135, 134, 137, 152, 165, 158, 141, 149, 105, 137, 167, 163, 165, 163, 164, 167, 169, 169, 169, 170, 173, 177, 178, 179, 180, 180, 181, 183, 184, 184, 186},
                {189, 190, 190, 178, 149, 148, 146, 153, 142, 134, 140, 147, 143, 141, 149, 134, 135, 138, 138, 135, 139, 142, 130, 137, 143, 145, 138, 139, 128, 124, 135, 143, 144, 146, 147, 147, 150, 153, 153, 150, 159, 118, 125, 169, 164, 165, 164, 164, 167, 168, 169, 169, 171, 174, 176, 177, 180, 181, 181, 182, 185, 186, 186, 187},
                {189, 190, 194, 178, 158, 162, 160, 160, 147, 142, 147, 160, 156, 160, 157, 143, 135, 150, 161, 159, 159, 145, 124, 128, 151, 156, 156, 153, 124, 123, 141, 155, 158, 158, 160, 162, 161, 160, 161, 163, 169, 135, 114, 170, 163, 164, 164, 165, 167, 169, 170, 170, 172, 174, 176, 177, 180, 181, 181, 182, 185, 186, 186, 187},
                {190, 188, 190, 172, 160, 165, 166, 162, 123, 125, 139, 158, 159, 160, 138, 115, 124, 146, 159, 153, 157, 130, 110, 121, 146, 154, 152, 147, 116, 120, 134, 149, 154, 155, 157, 157, 159, 162, 162, 160, 167, 146, 105, 167, 162, 165, 164, 167, 167, 169, 172, 172, 174, 176, 177, 177, 179, 181, 181, 182, 185, 186, 185, 186},
                {193, 189, 190, 171, 163, 162, 164, 152, 120, 131, 150, 164, 161, 160, 133, 113, 120, 145, 159, 152, 163, 130, 111, 122, 144, 156, 152, 139, 123, 125, 135, 151, 156, 156, 158, 159, 159, 159, 161, 163, 163, 155, 105, 162, 164, 166, 164, 167, 166, 168, 172, 173, 175, 178, 179, 179, 180, 181, 181, 182, 186, 187, 187, 189},
                {186, 186, 186, 168, 166, 159, 162, 144, 123, 132, 146, 157, 159, 159, 133, 118, 128, 150, 159, 153, 158, 129, 109, 119, 138, 153, 151, 136, 120, 121, 132, 151, 154, 149, 150, 156, 161, 160, 160, 163, 162, 163, 112, 153, 167, 165, 165, 166, 166, 169, 171, 173, 176, 179, 180, 180, 180, 182, 181, 182, 186, 188, 190, 193},
                {188, 188, 185, 166, 165, 158, 166, 142, 125, 134, 149, 159, 159, 157, 129, 117, 125, 148, 158, 153, 151, 129, 112, 119, 141, 151, 152, 142, 114, 118, 132, 148, 148, 151, 154, 155, 157, 159, 160, 159, 161, 165, 114, 142, 166, 163, 165, 166, 166, 169, 171, 172, 175, 178, 180, 179, 179, 181, 180, 182, 185, 188, 191, 194},
                {186, 186, 186, 166, 165, 158, 165, 137, 123, 134, 152, 158, 162, 155, 129, 115, 126, 149, 159, 150, 153, 126, 112, 119, 138, 152, 152, 140, 114, 114, 130, 146, 150, 149, 153, 158, 160, 159, 161, 165, 164, 168, 123, 129, 170, 162, 163, 165, 166, 168, 170, 170, 172, 175, 177, 177, 177, 180, 180, 181, 184, 188, 194, 199},
                {188, 189, 183, 164, 162, 154, 161, 132, 122, 134, 151, 159, 159, 155, 124, 113, 124, 146, 157, 153, 156, 127, 112, 120, 138, 150, 150, 141, 116, 116, 131, 147, 151, 150, 152, 155, 158, 158, 160, 162, 162, 168, 133, 123, 170, 163, 164, 161, 166, 168, 170, 170, 171, 173, 174, 174, 178, 180, 180, 182, 185, 188, 194, 200},
                {187, 190, 178, 163, 162, 152, 159, 129, 122, 137, 149, 158, 155, 153, 122, 113, 128, 149, 157, 156, 156, 126, 108, 120, 143, 151, 150, 144, 118, 115, 129, 146, 153, 152, 151, 153, 155, 156, 159, 159, 159, 164, 142, 111, 167, 164, 167, 164, 164, 166, 168, 169, 170, 171, 171, 171, 173, 175, 177, 181, 185, 189, 196, 204},
                {183, 186, 170, 160, 159, 149, 154, 122, 120, 138, 154, 161, 160, 153, 121, 112, 130, 151, 157, 157, 156, 129, 110, 126, 147, 151, 150, 146, 120, 112, 126, 144, 152, 152, 152, 152, 154, 157, 159, 159, 160, 164, 155, 112, 163, 164, 165, 165, 165, 167, 169, 170, 170, 170, 170, 170, 171, 172, 173, 177, 181, 182, 189, 198},
                {182, 184, 166, 159, 156, 147, 150, 116, 118, 139, 162, 166, 166, 153, 120, 109, 132, 155, 159, 158, 157, 130, 109, 125, 146, 149, 148, 147, 120, 110, 123, 144, 150, 152, 154, 154, 155, 157, 160, 160, 162, 163, 165, 121, 155, 160, 154, 156, 154, 154, 155, 155, 154, 154, 155, 155, 157, 158, 159, 162, 165, 166, 171, 179},
                {175, 174, 160, 158, 152, 149, 149, 116, 122, 142, 162, 161, 164, 149, 118, 109, 132, 155, 159, 157, 158, 127, 103, 119, 146, 149, 149, 148, 120, 109, 120, 142, 148, 151, 154, 154, 154, 156, 159, 160, 161, 159, 167, 126, 143, 161, 154, 159, 161, 160, 161, 161, 161, 161, 163, 164, 166, 167, 168, 171, 174, 176, 180, 186},
                {156, 155, 149, 153, 147, 148, 146, 112, 128, 144, 156, 152, 154, 145, 117, 111, 128, 149, 155, 155, 161, 127, 105, 122, 147, 149, 149, 149, 121, 108, 118, 139, 148, 149, 151, 152, 152, 154, 157, 159, 163, 158, 168, 128, 132, 167, 166, 168, 162, 162, 163, 164, 165, 165, 166, 168, 171, 172, 171, 172, 174, 174, 175, 179},
                {146, 145, 146, 154, 147, 150, 144, 109, 128, 143, 151, 151, 152, 147, 115, 110, 132, 149, 154, 154, 163, 125, 107, 127, 145, 147, 147, 149, 122, 109, 119, 139, 149, 148, 149, 150, 150, 152, 156, 158, 162, 157, 169, 126, 123, 167, 167, 163, 164, 163, 163, 164, 164, 163, 164, 166, 168, 170, 169, 169, 171, 172, 173, 175}
        };

        //**************** 读取第一层卷积参数开始 *************
        String conv_file = "E:\\Java project\\CNN\\src\\conv2d_1.txt";
        String conv_bias_file = "E:\\Java project\\CNN\\src\\conv2d_1_bias.txt";
        double[][][] conv2d_1_1 = new double[5][5][1];
        double[][][] conv2d_1_2 = new double[5][5][1];
        double[] conv2d_1_1_bias = new double[1];
        double[] conv2d_1_2_bias = new double[1];

        ReadPara_conv_1(conv_file, conv2d_1_1, conv2d_1_2);
        ReadPara_conv_1_bias(conv_bias_file, conv2d_1_1_bias, conv2d_1_2_bias);

        ArrayList<double[][][]> conv1_para = new ArrayList<>();
        ArrayList<double[]> conv1_bias = new ArrayList<>();
        conv1_para.add(conv2d_1_1);
        conv1_para.add(conv2d_1_2);
        conv1_bias.add(conv2d_1_1_bias);
        conv1_bias.add(conv2d_1_2_bias);
        //**************** 读取第一层卷积参数结束 *************


        //**************** 读取第二层卷积参数开始 *************
        conv_file = "E:\\Java project\\CNN\\src\\conv2d_2.txt";
        conv_bias_file = "E:\\Java project\\CNN\\src\\conv2d_2_bias.txt";
        double[][][] conv2d_2_1 = new double[21][21][2];
        double[][][] conv2d_2_2 = new double[21][21][2];
        double[] conv2d_2_1_bias = new double[1];
        double[] conv2d_2_2_bias = new double[1];

        ReadPara_conv_2(conv_file, conv2d_2_1, conv2d_2_2);
        ReadPara_conv_1_bias(conv_bias_file, conv2d_2_1_bias, conv2d_2_2_bias);

        ArrayList<double[][][]> conv2_para = new ArrayList<>();
        ArrayList<double[]> conv2_bias = new ArrayList<>();
        conv2_para.add(conv2d_2_1);
        conv2_para.add(conv2d_2_2);
        conv2_bias.add(conv2d_2_1_bias);
        conv2_bias.add(conv2d_2_2_bias);
        //**************** 读取第二层卷积参数结束 *************

        //**************** 读取 dense_1 参数开始 *************
        String dense_file = "E:\\Java project\\CNN\\src\\dense_1.txt";
        String dense_bias_file = "E:\\Java project\\CNN\\src\\dense_1_bias.txt";

        double[][] dense_1 = new double[968][128];
        double[] dense_1_bias = new double[128];

        ReadPara_dense(dense_file, dense_1);
        ReadPara_dense_bias(dense_bias_file, dense_1_bias);
        //**************** 读取 dense_1 参数结束 *************

        //**************** 读取 dense_2 参数开始 *************
        String dense_2_file = "E:\\Java project\\CNN\\src\\dense_2.txt";
        String dense_2_bias_file = "E:\\Java project\\CNN\\src\\dense_2_bias.txt";
        double[][] dense_2 = new double[128][4];
        double[] dense_2_bias = new double[4];

        ReadPara_dense(dense_2_file, dense_2);
        ReadPara_dense_bias(dense_2_bias_file, dense_2_bias);
//        util.printPara(dense_2);
        //**************** 读取 dense_2 参数结束 *************


        Tensor inputImage = new Tensor(imageData);

        CONV conv1 = new CONV(inputImage, 2, 5, "same");
        conv1.setParameters(conv1_para, conv1_bias);
        conv1.run_forward();
        conv1.print_output();
        Tensor conv1_out = util.Activation(conv1.getOutputTensor(), "tanh");

        CONV conv2 = new CONV(conv1_out, 2, 21, "full");
        conv2.setParameters(conv2_para, conv2_bias);
        conv2.run_forward();
        conv2.print_output();
        Tensor conv2_out = util.Activation(conv2.getOutputTensor(), "tanh");


        POOL pool1 = new POOL(conv2_out, POOL.POOL_TYPE.max, 2);
        pool1.run_forward();
        pool1.print_output();

        FLATTEN flatten1 = new FLATTEN(pool1.getOutputTensor());
        flatten1.run_forward();
        flatten1.print_output();

        DENSE dense1 = new DENSE(flatten1.getOutputTensor(), 128);
        dense1.setDensePara(dense_1, dense_1_bias);
        dense1.run_forward();
        dense1.print_output();
        Tensor dense1_out = util.Activation(dense1.getOutputTensor(), "tanh");

        DENSE dense2 = new DENSE(dense1.getOutputTensor(), 4);
        dense2.setDensePara(dense_2, dense_2_bias);
        dense2.run_forward();
        dense2.print_output();

        double[] result = util.softmax(util.T(dense2.getOutputTensor().getTensor().get(0)));
        for (double res : result) {
            System.out.println(res);
        }

        String imagepath = "E:\\Java project\\CNN\\src\\left.jpg";
        String outimage = "E:\\Java project\\CNN\\src\\out.jpg";
//        ReadPic("E:\\Java project\\CNN\\src\\left.jpg");
//        int[][] picdata = getPicData("E:\\Java project\\CNN\\src\\left.jpg");
//        getMatrixGray("E:\\Java project\\CNN\\src\\left1-1.jpg");
//        byte[] res = util.image2byte(imagepath);
//        util.byte2image(res, outimage);


        System.out.println("\n END!!!  " + imageData[0].length);
    }


    public static void ReadPara_conv_1(String parafile, double[][][] conv2d_1_1, double[][][] conv2d_1_2) {

//        ArrayList<String> temp = util.ReadTxt("E:\\Java project\\CNN\\src\\conv2d_1.txt");
        ArrayList<String> temp = util.ReadTxt(parafile);
        System.out.println("**************" + temp.size());

        ArrayList<Double> result = new ArrayList<>();
        int i = 0;
        int j = 0;
        for (String line : temp) {
            // 如果为空，则跳出，发生在文件最后
            if (line == null) {
                break;
            }
            // 如果有空行，则跳过
            if ("".equals(line)) {
                continue;
            }
            // 去掉无用字符
            line = line.replace("[", "");
            line = line.replace("]", "");
            line = line.replace('\n', ' ');
//            line = line.replace(" ", "");
            // 讲字符字符串按照空格进行划分
            String[] splited = line.split("\\s+");

            try {
                for (String s : splited) {
                    if (s.equals('\n')) {
                        System.out.println("@@@");
                    } else if (s.equals("")) {            // 如果有空格，跳过
                        continue;
                    }
                    result.add(Double.parseDouble(s));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(result.size());
        for (int k = 0; k < result.size() / 2; k++) {
            conv2d_1_1[k / 5][k % 5][0] = result.get(k);
            conv2d_1_2[k / 5][k % 5][0] = result.get(k + 25);
        }
    }

    public static void ReadPara_conv_1_bias(String parafile, double[] conv2d_1_1_bias, double[] conv2d_1_2_bias) {

//        ArrayList<String> temp = util.ReadTxt("E:\\Java project\\CNN\\src\\conv2d_1_bias.txt");
        ArrayList<String> temp = util.ReadTxt(parafile);
        System.out.println("**************" + temp.size());

        ArrayList<Double> result = new ArrayList<>();
        for (String line : temp) {
            // 如果为空，则跳出，发生在文件最后
            if (line == null) {
                break;
            }
            // 如果有空行，则跳过
            if ("".equals(line)) {
                continue;
            }
            // 去掉无用字符
            line = line.replace("[", "");
            line = line.replace("]", "");
            line = line.replace('\n', ' ');
//            line = line.replace(" ", "");
            // 讲字符字符串按照空格进行划分
            String[] splited = line.split("\\s+");

            try {
                for (String s : splited) {
                    if (s.equals('\n')) {
                        System.out.println("@@@");
                    } else if (s.equals("")) {            // 如果以后空格，跳过
                        continue;
                    }
                    result.add(Double.parseDouble(s));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(result.size());

        conv2d_1_1_bias[0] = result.get(0);
        conv2d_1_2_bias[0] = result.get(1);

    }

    public static void ReadPara_conv_2(String parafile, double[][][] conv2d_2_1, double[][][] conv2d_2_2) {

//        ArrayList<String> temp = util.ReadTxt("E:\\Java project\\CNN\\src\\conv2d_1.txt");
        ArrayList<String> temp = util.ReadTxt(parafile);
        System.out.println("the size befor strip: " + temp.size());

        ArrayList<Double> result = new ArrayList<>();
        int i = 0;
        int j = 0;
        for (String line : temp) {
            // 如果为空，则跳出，发生在文件最后
            if (line == null) {
                break;
            }
            // 如果有空行，则跳过
            if ("".equals(line)) {
                continue;
            }
            // 去掉无用字符
            line = line.replace("[", "");
            line = line.replace("]", "");
            line = line.replace('\n', ' ');
//            line = line.replace(" ", "");
            // 讲字符字符串按照空格进行划分
            String[] splited = line.split("\\s+");

            try {
                for (String s : splited) {
                    if (s.equals('\n')) {
                        System.out.println("@@@");
                    } else if (s.equals("")) {            // 如果有空格，跳过
                        continue;
                    }
                    result.add(Double.parseDouble(s));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("the size after strip: " + result.size());

        for (int k = 0; k < result.size() / 2; k += 2) {
            // 下面的index需要仔细推导
            conv2d_2_1[k / 42][k % 42 / 2][0] = result.get(k);
            conv2d_2_2[k / 42][k % 42 / 2][0] = result.get(k + 882);
            conv2d_2_1[k / 42][k % 42 / 2][1] = result.get(k + 1);
            conv2d_2_2[k / 42][k % 42 / 2][1] = result.get(k + 882 + 1);

        }
    }

    public static void ReadPara_dense(String parafile, double[][] dense_1) {

//        ArrayList<String> temp = util.ReadTxt("E:\\Java project\\CNN\\src\\conv2d_1.txt");
        ArrayList<String> temp = util.ReadTxt(parafile);
        System.out.println("the size befor strip: " + temp.size());

        ArrayList<Double> result = new ArrayList<>();
        for (String line : temp) {
            // 如果为空，则跳出，发生在文件最后
            if (line == null) {
                break;
            }
            // 如果有空行，则跳过
            if ("".equals(line)) {
                continue;
            }
            // 去掉无用字符
            line = line.replace("[", "");
            line = line.replace("]", "");
            line = line.replace('\n', ' ');
//            line = line.replace(" ", "");
            // 讲字符字符串按照空格进行划分
            String[] splited = line.split("\\s+");

            try {
                for (String s : splited) {
                    if (s.equals('\n')) {
                        System.out.println("@@@");
                    } else if (s.equals("")) {            // 如果有空格，跳过
                        continue;
                    }
                    result.add(Double.parseDouble(s));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("the size after strip: " + result.size());
        int index = 0;
        for (int i = 0; i < dense_1.length; i++) {
            for (int j = 0; j < dense_1[0].length; j++) {
                dense_1[i][j] = result.get(index);
                index++;
            }
        }
    }

    public static void ReadPara_dense_bias(String parafile, double[] dense_bias) {

        ArrayList<String> temp = util.ReadTxt(parafile);
        System.out.println("**************" + temp.size());

        ArrayList<Double> result = new ArrayList<>();
        for (String line : temp) {
            // 如果为空，则跳出，发生在文件最后
            if (line == null) {
                break;
            }
            // 如果有空行，则跳过
            if ("".equals(line)) {
                continue;
            }
            // 去掉无用字符
            line = line.replace("[", "");
            line = line.replace("]", "");
            line = line.replace('\n', ' ');
            // 讲字符字符串按照空格进行划分
            String[] splited = line.split("\\s+");

            try {
                for (String s : splited) {
                    if (s.equals('\n')) {
                        System.out.println("@@@");
                    } else if (s.equals("")) {            // 如果以后空格，跳过
                        continue;
                    }
                    result.add(Double.parseDouble(s));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(result.size());
        int index = 0;
        for (int i = 0; i < dense_bias.length; i++) {
            dense_bias[i] = result.get(index);
            index++;
        }
    }

    public static void ReadPic(String picpath) {
        try {
            FileInputStream fis = new FileInputStream(new File(picpath));
            FileOutputStream fos = new FileOutputStream(new File("E:\\Java project\\CNN\\src\\test.jpg"));
            byte[] read = new byte[1024];
            int len = 0;
            while ((len = fis.read(read)) != -1) {
                fos.write(read, 0, len);
            }
            for (byte d :
                    read) {
                System.out.println(d);
            }
            System.out.println(read.length);

            fis.close();
            fos.flush();
            fos.close();
        } catch (FileNotFoundException e) {
            System.out.println("PICTURE is not found! ");
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public static byte[] getMatrixGray(String imagepath) {

        try {
            BufferedImage image = ImageIO.read(new File(imagepath));
            FileOutputStream fos = new FileOutputStream(new File("E:\\Java project\\CNN\\src\\test.jpg"));
            // 转灰度图像
            BufferedImage grayImage = new BufferedImage(image.getWidth(), image.getHeight(),
                    BufferedImage.TYPE_BYTE_GRAY);
            new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(image, grayImage);
            // getData方法返回BufferedImage的raster成员对象
            // 经检验，res内存的数据为 64*64 byte
            byte[] res = (byte[]) grayImage.getData().getDataElements(0, 0, image.getWidth(), image.getHeight(), null);
            byte[][] out = new byte[image.getWidth()][image.getHeight()];
            int index = 0;
            for (int i = 0; i < res.length; i++) {
                res[i] += 128;
            }
            FileOutputStream fout = new FileOutputStream("E:\\Java project\\CNN\\src\\test.jpg");
            //将字节写入文件
            fout.write(res);
            fout.close();

            return res;
        } catch (IOException e) {
            e.printStackTrace();
        } finally {

        }
        return null;
    }


}
