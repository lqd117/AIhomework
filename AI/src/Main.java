import java.awt.RenderingHints.Key;
import java.beans.Expression;
import java.util.HashMap;
import java.util.Map;
import java.lang.Math;
import java.math.MathContext;
import java.time.Year;

import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;



public class Main {
	
	public static void main(String[] args){
//	    String expression = "math.sqrt(x)";
//	    MyExpression.init(expression);
//	    double y = MyExpression.getYByX(9);
//	    System.out.println(y);
		
//		NumpyRandom rand = new NumpyRandom();
//		rand.setSeed(3);
//		for (int i = 0; i < 10; i++)
//			System.out.println(" " + rand.nextGaussian());
		
		MyNeuralNetwork myNeuralNetwork = new MyNeuralNetwork();
		int[] a = new int[3];
		a[0]=5;a[1]=4;a[2]=3;
		Map<String, MyArray> parameters = myNeuralNetwork.initializeParametersDeep(a);
		for (String key : parameters.keySet()) {
			System.out.println(key);
			parameters.get(key).display();
		}
		
		MyArray A = new MyArray(3, 2);
		A.set(0, 0, 1.62434536);A.set(0, 1, -0.61175641);
		A.set(1, 0, -0.52817175);A.set(1, 1, -1.07296862);
		A.set(2, 0, 0.86540763);A.set(2, 1, -2.3015387);
		MyArray W = new MyArray(1, 3);
		W.set(0, 0, 1.74481176);W.set(0, 1, -0.7612069);W.set(0, 2, 0.3190391);
		MyArray b = new MyArray(1, 1);
		b.set(0, 0, -0.24937038);
		
		TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> twoTuple = 
				myNeuralNetwork.linear_forward(A, W, b);
		MyArray Z = twoTuple.getFirst();
		Z.display();
		
	}
}


