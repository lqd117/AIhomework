import java.util.ArrayList;
import java.util.Map;

import org.apache.commons.math3.analysis.function.Multiply;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;

public class Controller {
	private MyNeuralNetwork myNeuralNetwork = new MyNeuralNetwork();
	private String expression;
	private MyExpression myExpression = new MyExpression();
	private int dataNumber,batchNumber,learning_rate;//样本数，迭代数
	private MyArray data, Y;
	private int[] layers;
	private Map<String, MyArray> parameters;
	
	public Controller(String tempExpression,int tempDataNumber,int tempBatchNUmber, int rate,int[] tempLayers){
		this.learning_rate = rate;
		this.layers = tempLayers;
		this.dataNumber = tempDataNumber;
		this.batchNumber = tempBatchNUmber;
		this.expression = tempExpression;
		this.myExpression.init(this.expression);
		this.data = new MyArray(1, this.dataNumber);
		this.Y = new MyArray(1, this.dataNumber);
		this.createData();
		this.parameters = this.myNeuralNetwork.initialize_parameters_deep(this.layers);
	}
	
	public MyArray test(MyArray x){
		TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>> temp;
		temp = this.myNeuralNetwork.L_model_forward(this.data, this.parameters);
		return temp.getFirst();
	}
	
	public void runOneBatch(){
		TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>> temp;
		ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>> caches;
		Map<String, MyArray> grads;
		MyArray AL;
		double cost;
		
		temp = this.myNeuralNetwork.L_model_forward(this.data, this.parameters);
		AL = temp.getFirst();
		caches = temp.getSecond();
		cost = this.myNeuralNetwork.compute_cost(AL, this.Y);
		System.out.println(cost);
		grads = this.myNeuralNetwork.L_model_backward(AL, this.Y, caches);
		this.parameters = this.myNeuralNetwork.update_parameters(this.parameters, grads, this.learning_rate);
			
	}
	
	public void createData(){
		for(int i=0;i<this.data.getColumns();i++){
			double x = Math.random()*5;
			this.data.set(0, i, x);
			this.Y.set(0, i, this.myExpression.getYByX(x));
		}
	}

	
}
