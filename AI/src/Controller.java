import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.analysis.function.Multiply;
import org.omg.PortableServer.THREAD_POLICY_ID;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;

public class Controller {
	private MyNeuralNetwork myNeuralNetwork = new MyNeuralNetwork();
	private String expression;
	private MyExpression myExpression = new MyExpression();
	private int dataNumber,batchNumber;//样本数，迭代数
	private double learning_rate;
	public MyArray data, Y;
	private int[] layers;
	private Map<String, MyArray> parameters = new HashMap<>();
	
	public Controller(String tempExpression,int tempDataNumber,int tempBatchNUmber, double rate,int[] tempLayers){
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
	
	public double runOneBatch(){
		TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>> temp;
		Map<String, MyArray> grads;
		double cost;
		temp = this.myNeuralNetwork.L_model_forward(this.data, this.parameters);
		cost = this.myNeuralNetwork.compute_cost(temp.getFirst(), this.Y);
		grads = this.myNeuralNetwork.L_model_backward(temp.getFirst(), this.Y, temp.getSecond());
		this.parameters = this.myNeuralNetwork.update_parameters(this.parameters, grads, this.learning_rate);
		
		
		return cost;
	}
	
	public void createData(){
		for(int i=0;i<this.data.getColumns();i++){
			double x = ((double)i-(double)this.data.getColumns()/2)/((double)this.data.getColumns()/4);
			this.data.set(0, i, x);
			this.Y.set(0, i, this.myExpression.getYByX(x));
		}
		this.data.display();
		this.Y.display();
	}

	
}
