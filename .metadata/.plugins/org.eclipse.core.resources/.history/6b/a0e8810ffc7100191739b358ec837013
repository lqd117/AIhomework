import java.lang.reflect.Array;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.exception.ZeroException;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;

public class MyNeuralNetwork {
	
	private static MyNumPy np = new MyNumPy();
	
	public static Map<String, MyArray> initializeParametersDeep(int[] layer_dims){
		/**
		 * Arguments:
		 * layer_dims -- array (list) containing the dimensions of each layer in our network
		 * 
		 * Returns:
		 * parameters --  dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
		 * 		Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
		 * 		bl -- bias vector of shape (layer_dims[l], 1)
		 */
		np.random.setSeed(3);
		Map<String, MyArray> parameters = new HashMap<>();
		int L = layer_dims.length;
		for(int i=1;i<L;i++){
			parameters.put("W"+String.valueOf(i), np.random.randn(layer_dims[i],layer_dims[i-1]).mult(0.1));
			parameters.put("b"+String.valueOf(i), np.zeros(layer_dims[i],1));
		}
		return parameters;
	}

	public static TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> 
									linear_forward(MyArray A,MyArray W,MyArray b){
		
		/**
		 * Implement the linear part of a layer's forward propagation.
		 * Arguments:
		 * A -- activations from previous layer (size of previous layer, number of examples)
		 * W -- weights matrix(size of current layer, size of previous layer)
		 * b -- bias vector (size of the current layer, 1)
		 * 
		 * Returns:
		 * Z -- the input of the activation function, also called pre-activation parameter 
		 * cache -- a dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
		 */
		MyArray Z = np.dot(W,A).add(b);
		ThreeTuple<MyArray, MyArray, MyArray> cache = 
				new ThreeTuple<MyArray, MyArray, MyArray>(A, W, b);
		TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> twoTuple = 
				new TwoTuple<MyArray, ThreeTuple<MyArray,MyArray,MyArray>>(Z, cache);
		
		return twoTuple;
	}
}
