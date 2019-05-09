import java.awt.RenderingHints.Key;
import java.beans.Expression;
import java.util.ArrayList;
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
//		int[] a = new int[3];
//		a[0]=5;a[1]=4;a[2]=3;
//		Map<String, MyArray> parameters = myNeuralNetwork.initialize_parameters_deep(a);
//		for (String key : parameters.keySet()) {
//			System.out.println(key);
//			parameters.get(key).display();
//		}
		
//		MyArray A = new MyArray(3, 2);
//		A.set(0, 0, 1.62434536);A.set(0, 1, -0.61175641);
//		A.set(1, 0, -0.52817175);A.set(1, 1, -1.07296862);
//		A.set(2, 0, 0.86540763);A.set(2, 1, -2.3015387);
//		MyArray W = new MyArray(1, 3);
//		W.set(0, 0, 1.74481176);W.set(0, 1, -0.7612069);W.set(0, 2, 0.3190391);
//		MyArray b = new MyArray(1, 1);
//		b.set(0, 0, -0.24937038);
//		
//		TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> twoTuple = 
//				myNeuralNetwork.linear_forward(A, W, b);
//		MyArray Z = twoTuple.getFirst();
//		Z.display();
		
//		MyArray A_prev = new MyArray(3, 2);
//		A_prev.set(0, 0, -0.41675785);A_prev.set(0, 1, -0.05626683);
//		A_prev.set(1, 0, -2.1361961);A_prev.set(1, 1, 1.64027081);
//		A_prev.set(2, 0, -1.79343559);A_prev.set(2, 1, -0.84174737);
//		MyArray W = new MyArray(1, 3);
//		W.set(0, 0, 0.50288142);W.set(0, 1, -1.24528809);W.set(0, 2, -1.05795222);
//		MyArray b = new MyArray(1, 1);
//		b.set(0, 0, -0.90900761);
//		TwoTuple<MyArray, TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>> twoTuple = 
//				myNeuralNetwork.linear_activation_forward(A_prev, W, b, "sigmoid");
//		MyArray A = twoTuple.getFirst();
//		A.display();
//		
//		twoTuple = myNeuralNetwork.linear_activation_forward(A_prev, W, b, "relu");
//		A = twoTuple.getFirst();
//		A.display();
		
//		MyArray X = new MyArray(5, 4);
//		X.set(0, 0, -0.31178367);X.set(0, 1, 0.72900392);X.set(0, 2, 0.21782079);X.set(0, 3, -0.8990918);
//		X.set(1, 0, -2.48678065);X.set(1, 1, 0.91325152);X.set(1, 2, 1.12706373);X.set(1, 3, -1.51409323);
//		X.set(2, 0, 1.63929108);X.set(2, 1, -0.4298936);X.set(2, 2, 2.63128056);X.set(2, 3, 0.60182225);
//		X.set(3, 0, -0.33588161);X.set(3, 1, 1.23773784);X.set(3, 2, 0.11112817);X.set(3, 3, 0.12915125);
//		X.set(4, 0, 0.07612761);X.set(4, 1, -0.15512816);X.set(4, 2, 0.63422534);X.set(4, 3, 0.810655);
//		
//		MyArray W1 = new MyArray(4, 5);
//		W1.set(0, 0, 0.35480861);W1.set(0, 1, 1.81259031);W1.set(0, 2, -1.3564758);W1.set(0, 3, -0.46363197);W1.set(0, 4, 0.82465384);
//		W1.set(1, 0, -1.17643148);W1.set(1, 1, 1.56448966);W1.set(1, 2, 0.71270509);W1.set(1, 3, -0.1810066);W1.set(1, 4, 0.53419953);
//		W1.set(2, 0, -0.58661296);W1.set(2, 1, -1.48185327);W1.set(2, 2, 0.85724762);W1.set(2, 3, 0.94309899);W1.set(2, 4, 0.11444143);
//		W1.set(3, 0, -0.02195668);W1.set(3, 1, -2.12714455);W1.set(3, 2, -0.83440747);W1.set(3, 3, -0.46550831);W1.set(3, 4, 0.23371059);
//		
//		MyArray b1 = new MyArray(4, 1);
//		b1.set(0, 0, 1.38503523);
//		b1.set(1, 0, -0.51962709);
//		b1.set(2, 0, -0.78015214);
//		b1.set(3, 0, 0.95560959);
//		
//		MyArray W2 = new MyArray(3, 4);
//		W2.set(0, 0, -0.12673638);W2.set(0, 1, -1.36861282);W2.set(0, 2, 1.21848065);W2.set(0, 3, -0.85750144);
//		W2.set(1, 0, -0.56147088);W2.set(1, 1, -1.0335199);W2.set(1, 2, 0.35877096);W2.set(1, 3, 1.07368134);
//		W2.set(2, 0, -0.37550472);W2.set(2, 1, 0.39636757);W2.set(2, 2, -0.47144628);W2.set(2, 3, 2.33660781);
//		
//		MyArray b2 = new MyArray(3, 1);
//		b2.set(0, 0, 1.50278553);
//		b2.set(1, 0, -0.59545972);
//		b2.set(2, 0, 0.52834106);
//		
//		MyArray W3 = new MyArray(1, 3);
//		W3.set(0, 0, 0.9398248);W3.set(0, 1, 0.42628539);W3.set(0, 2, -0.75815703);
//		
//		MyArray b3 = new MyArray(1, 1);
//		b3.set(0, 0, -0.16236698);
//		
//		Map<String, MyArray> parameters = new HashMap<>();
//		parameters.put("W1", W1);parameters.put("b1", b1);
//		parameters.put("W2", W2);parameters.put("b2", b2);
//		parameters.put("W3", W3);parameters.put("b3", b3);
//		
//		TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>> twoTuple = 
//				myNeuralNetwork.L_model_forward(X, parameters);
//		
//		MyArray AL;
//		AL = twoTuple.getFirst();
//		AL.display();
//		System.out.println(twoTuple.getSecond().size());
		
//		MyArray Y = new MyArray(1, 3);
//		Y.set(0, 0, 1);Y.set(0, 1, 1);Y.set(0, 2, 1);
//		MyArray AL = new MyArray(1, 3);
//		AL.set(0, 0, 0.8);AL.set(0, 1, 0.9);AL.set(0, 2, 0.4);
//		double cost = myNeuralNetwork.compute_cost(AL, Y);
//		System.out.println(cost);
		
	}
}



