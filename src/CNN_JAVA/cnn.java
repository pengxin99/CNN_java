package CNN_JAVA;

import CNN_JAVA.util.*;
import sun.rmi.server.Activation;

interface Layer {
    enum LayerType {
        // 网络层的类型：输入层、输出层、卷积层、采样层
        input, output, conv, samp
    }

    void run_forward();
    void print_output();
    Tensor getOutputTensor() ;
}

class Tensor {
    private double[][] tensor;

    Tensor() {
    }

    Tensor(double[][] tensor) {
        this.tensor = tensor;
    }

    Tensor(int height, int width) {
        this.tensor = new double[height][width];
    }

    public double[][] getTensor() {
        return this.tensor;
    }

    public void setTensorByPixel(int i, int j, double val) {
        this.tensor[i][j] = val;
    }

    public void printTensor() {
        for (int i = 0; i < tensor.length; i++) {
            for (int j = 0; j < tensor[0].length; j++) {
                System.out.print(this.tensor[i][j] + "\t");
            }
            System.out.println();
        }
    }

    public double[][] T(){
        int x = this.tensor.length;
        int y= this.tensor[0].length;
        double[][] temp = new double[y][x];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[j][i] = this.tensor[i][j];
            }
        }
        return temp;
    }
}

class CONV implements Layer {
    private LayerType Conv;
    private int FilterNum;
    private int FilterSize;
    private double[][][] FilterPara = new double[FilterNum][FilterSize][FilterSize];
    private double[] Bias = new double[FilterNum];      // 每个卷积滤波器设置一个bias
    private Tensor inputTensor;
    private Tensor outputTensor;

    public CONV(Tensor inputTensor, int filternum, int filtersize) {
        this.inputTensor = inputTensor;
        this.FilterNum = filternum;
        this.FilterSize = filtersize;
    }

    public void setParameters(double[][][] para, double[] bias) {
        this.FilterPara = para;
        this.Bias = bias;
    }
    public void printPara(){
        for (int i = 0; i < this.FilterPara.length; i++) {
            for (int j = 0; j < this.FilterPara[0].length; j++) {
                for (int k = 0; k < this.FilterPara[0][0].length; k++) {
                    System.out.println(this.FilterPara[i][j][k]);
                }
            }
        }
    }
    public Tensor getOutputTensor(){
        return this.outputTensor;
    }
    @Override
    public void run_forward() {
        int InputWidth = this.inputTensor.getTensor().length;
        int InputHeight = this.inputTensor.getTensor()[0].length;
        int OutputWidth = InputWidth - this.FilterSize + 1;
        int OutputHeight = InputHeight - this.FilterSize + 1;
        // 设定输出向量的临时值
        Tensor output_temp = new Tensor(OutputHeight, OutputWidth);
        // 对所有的filter进行卷积计算
        for (int filter = 0; filter < FilterNum; filter++) {
            // filter在输入tensor上滑动
            for (int j = 0; j < OutputWidth; j++) {
                for (int i = 0; i < OutputHeight; i++) {
                    // 每次滑动计算生成的单个神经元的值
                    double temp = 0.0;
                    for (int k = 0; k < this.FilterSize; k++) {
                        for (int l = 0; l < this.FilterSize; l++) {
                            temp += this.inputTensor.getTensor()[i + k][j + l] * FilterPara[filter][k][l] ;
                        }
                    }
                    // 记录单次滑动卷积层生成的值
                    output_temp.setTensorByPixel(i, j, temp+this.Bias[filter]);
                }
            }
            this.outputTensor = output_temp;
        }

    }

    @Override
    public void print_output() {
        System.out.println("****** After the Conv layer");
        outputTensor.printTensor();
    }
}

class POOL implements Layer{
    enum POOL_TYPE{max,average}
    private POOL_TYPE type ;
    private int PoolSize ;
    private Tensor InputTensor ;
    private Tensor OutputTensor;

    public POOL(){}
    public POOL(Tensor input, POOL_TYPE type, int poolSize){
        this.InputTensor = input;
        this.type = type;
        this.PoolSize = poolSize;
    }

    public void padding(){
        int size = this.InputTensor.getTensor().length ;
        // 如果输入tensor的shape不是偶数，则进行增加一行，一列的padding操作
        if (size / 2 != 0){
            Tensor temp_tensor = new Tensor(size+1,size+1);
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    temp_tensor.setTensorByPixel(i,j,this.InputTensor.getTensor()[i][j]);
                }
            }
            // padding新加的一行和一列
            for (int i = 0; i < size+1; i++) {
                temp_tensor.setTensorByPixel(i,size,0);
                temp_tensor.setTensorByPixel(size,i,0);
            }
            this.InputTensor = temp_tensor;
        }
    }

    @Override
    public void run_forward() {
        // 在前向传播之前先进行padding
//        padding();
        int inputsize = InputTensor.getTensor().length;
        int outputsize = inputsize / this.PoolSize ;
        OutputTensor = new Tensor(outputsize,outputsize) ;
        int x = 0;
        int y = 0;
        for (int i = 0; i < inputsize; i+=this.PoolSize) {
            // 如果这次循环结束时越界，则舍弃
            if(i+this.PoolSize-1>=inputsize)
                break;
            for (int j = 0; j < inputsize; j+=this.PoolSize) {
                // 如果这次循环结束时越界，则舍弃
                if (j+this.PoolSize-1>=inputsize)
                    break;
                double temp = findMax(i,j);
                OutputTensor.setTensorByPixel(x,y,temp);
                // 一次输入到outputTensor中，以x为索引
                x++;
            }
            // 开始新一的行，x置0，逐行扫描
            x = 0;
            y++;
        }
    }

    @Override
    public void print_output() {
        System.out.println("****** After the Pooling layer");
        OutputTensor.printTensor();
    }

    public double findMax(int x, int y){
        double temp_max = 0 ;
        for (int i = x; i < x+this.PoolSize; i++) {
            for (int j = y; j < y+this.PoolSize; j++) {
                if (this.InputTensor.getTensor()[i][j] > temp_max){
                    temp_max = this.InputTensor.getTensor()[i][j];
                }
            }
        }
        return temp_max;
    }

    public Tensor getOutputTensor(){
        return this.OutputTensor;
    }
}

class FLATTEN implements Layer{
    private Tensor InputTensor;
    private Tensor OutputTensor;
    private int InputSize ;
    private int OutputSize ;

    public FLATTEN(){}

    public FLATTEN(Tensor input){
        this.InputTensor = input;
    }

    public Tensor getOutputTensor(){
        return this.OutputTensor;
    }
    @Override
    public void run_forward() {
        this.InputSize = InputTensor.getTensor().length;
        this.OutputSize = this.InputSize * this.InputSize;
        OutputTensor = new Tensor(this.OutputSize,1);
        int index = 0;
        for (int i = 0; i < this.InputSize; i++) {
            for (int j = 0; j < this.InputSize; j++) {
                OutputTensor.setTensorByPixel(index,0, InputTensor.getTensor()[i][j]);
                index ++;
            }
        }
    }

    @Override
    public void print_output() {
        System.out.println("****** After the Flatten layer");
        OutputTensor.printTensor();
    }
}

class DENSE implements Layer{
    private Tensor InputTensor;
    private Tensor OutputTensor;
    private double[][] DensePara ;
    private double[] DenseBias;

    public DENSE(){}
    public DENSE(Tensor input, int outsize ){
        this.InputTensor = input;
        this.OutputTensor = new Tensor(outsize, 1);
    }

    public void setDensePara(double[][] densePara, double[] denseBias) {
        this.DensePara = densePara;
        this.DenseBias = denseBias;
    }

    public Tensor getOutputTensor(){
        return this.OutputTensor;
    }
    @Override
    public void run_forward() {
        for (int i = 0; i < this.OutputTensor.getTensor().length; i++) {
            double temp = 0.0 ;
            for (int j = 0; j < this.InputTensor.getTensor().length; j++) {
                temp += this.DensePara[i][j] * this.InputTensor.getTensor()[j][0];
            }
            this.OutputTensor.setTensorByPixel(i,0,temp+this.DenseBias[i]);
        }
    }

    @Override
    public void print_output() {
        System.out.println("****** After the Dense layer");
        this.OutputTensor.printTensor();
    }
}



public class cnn {
    public static void main(String[] args) {
        // set image
        double[][] image = {{1.0, 1.0, 1.0, 1.0, 1.0,1.0},
                {1.0, 1.0, 5.0, 1.0, 1.0,1.0},
                {1.0, 1.0, 5.0, 1.0, 1.0,1.0},
                {1.0, 1.0, 5.0, 1.0, 1.0,1.0},
                {1.0, 1.0, 5.0, 1.0, 1.0,1.0},
                {1.0, 1.0, 5.0, 1.0, 1.0,1.0}};
        // set conv para
        double[][][] para = {{{0, 1.0, 0},
                            {0, 1.0, 0},
                            {0, 1.0, 0}}};
        double[] bias = {2.0};
        double[][][] uu = para;
        // set dense para
        double[][] densepara = {{1.0,1.0,1.0,1.0},
                {-1.0,-1.0,-1.0,-1.0}};
        double[] densebias = {0.0, 0.0} ;



        Tensor inputImage = new Tensor(image);

        CONV conv1 = new CONV(inputImage, 1, 2);
        conv1.setParameters(para,bias);
        conv1.run_forward();
        conv1.print_output();
        Tensor conv1_out = util.Activation(conv1.getOutputTensor(),"tanh");

        POOL pool1 = new POOL(conv1_out, POOL.POOL_TYPE.max,2) ;
        pool1.run_forward();
        pool1.print_output();

        FLATTEN flatten1 = new FLATTEN(pool1.getOutputTensor()) ;
        flatten1.run_forward();
        flatten1.print_output();

        DENSE dense1 = new DENSE(flatten1.getOutputTensor(), 2);
        dense1.setDensePara(densepara,densebias);
        dense1.run_forward();
        dense1.print_output();

        double[] result = util.softmax(dense1.getOutputTensor().T()[0]);
        for (double res:result) {
            System.out.println(res);
        }


        String packageName = cnn.class.getPackage().getName();
        System.out.println(packageName);

    }
}

