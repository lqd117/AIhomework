import java.lang.reflect.Array;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.analysis.function.Multiply;
import org.apache.commons.math3.exception.ZeroException;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;

public class MyNeuralNetwork {
	
	private static MyNumPy np = new MyNumPy();
	
	public static Map<String, MyArray> initialize_parameters_deep(int[] layer_dims){
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

	public static TwoTuple<MyArray, TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>
									linear_activation_forward(MyArray A_prev,MyArray W,
											MyArray b,String activation){
		/**
		 * Implement the forward propagation for the LINEAR->ACTIVATION layer
		 * 
		 * Arguments:
		 * A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
		 * W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
		 * b -- bias vector, numpy array of shape (size of the current layer, 1)
		 * activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
		 * 
		 * Returns:
		 * A -- the output of the activation function, also called the post-activation value 
		 * cache -- a dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
		 */
		MyArray Z,A = null,activation_cache = null;
		ThreeTuple<MyArray, MyArray, MyArray> linear_cache = null;
		if(activation == "sigmoid"){
			TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> twoTuple =
					linear_forward(A_prev, W, b);
			Z = twoTuple.getFirst();
			linear_cache = twoTuple.getSecond();
			TwoTuple<MyArray, MyArray> twoTuple2 = sigmoid(Z);
			A = twoTuple2.getFirst();
			activation_cache = twoTuple2.getSecond();
		}else if(activation == "relu"){
			TwoTuple<MyArray, ThreeTuple<MyArray, MyArray, MyArray>> twoTuple =
					linear_forward(A_prev, W, b);
			Z = twoTuple.getFirst();
			linear_cache = twoTuple.getSecond();
			TwoTuple<MyArray, MyArray> twoTuple2 = relu(Z);
			A = twoTuple2.getFirst();
			activation_cache = twoTuple2.getSecond();
		}
		TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray> cache = 
				new TwoTuple<ThreeTuple<MyArray,MyArray,MyArray>, MyArray>(linear_cache, activation_cache);
		TwoTuple<MyArray, TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>> twoTuple =
				new TwoTuple<MyArray, TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>(A,cache);
		return twoTuple;
	}
	
	public static TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>>
									L_model_forward(MyArray X, Map<String, MyArray> parameters) {
		/**
		 * Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
		 * 
		 * Arguments:
		 * X -- data, numpy array of shape (input size, number of examples)
		 * parameters -- output of initialize_parameters_deep()
		 * 
		 * Returns:
		 * AL -- last post-activation value
		 * caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
		 */
		ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>> caches =
				new ArrayList<>();
		MyArray A = X,A_prev,AL;
		TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray> cache;
		TwoTuple<MyArray, TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>> twoTuple;
		int L = parameters.size()/2;
		for(int i=1;i<L;i++){
			A_prev = new MyArray(A.getRows(), A.getColumns());
			for(int i1=0;i1<A_prev.getRows();i1++){
				for(int j1=0;j1<A_prev.getColumns();j1++){
					A_prev.set(i1, j1, A.getNumber(i1, j1));
				}
			}
			twoTuple = linear_activation_forward(
					A_prev, parameters.get("W"+String.valueOf(i)), parameters.get("b"+String.valueOf(i)), "relu");
			A = twoTuple.getFirst();
			cache = twoTuple.getSecond();
			caches.add(cache);
		}
		twoTuple = linear_activation_forward(
				A, parameters.get("W"+String.valueOf(L)), parameters.get("b"+String.valueOf(L)), "sigmoid");
		AL = twoTuple.getFirst();
		cache = twoTuple.getSecond();
		caches.add(cache);
		TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray, MyArray, MyArray>, MyArray>>> result =
				new TwoTuple<MyArray, ArrayList<TwoTuple<ThreeTuple<MyArray,MyArray,MyArray>,MyArray>>>(AL, caches);
		return result;
	}
	
	public static double compute_cost(MyArray AL,MyArray Y){
		/**
		 * Implement the cost function defined by equation (7).
		 * 
		 * Arguments:
		 * AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
		 * Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
		 * 
		 * Returns:
		 * cost -- cross-entropy cost
		 */
		int m = Y.getColumns();
		double cost=0;
		MyArray temp = np.dot(np.log(AL), Y.getT()).add(np.dot(np.log(AL.getOpposite().add(1)), Y.getOpposite().add(1).getT()));
		for(int i=0;i<temp.getRows();i++){
			for(int j=0;j<temp.getColumns();j++){
				cost += temp.getNumber(i, j);
			}
		}
		cost = -1.0/(double)m * cost;
		return cost;
	}
	
	
	public static TwoTuple<MyArray, MyArray> sigmoid(MyArray Z){
		/**
		 * Implements the sigmoid activation in numpy
		 * 
		 * Arguments:
		 * Z -- numpy array of any shape
		 * 
		 * Returns:
		 * A -- output of sigmoid(z), same shape as Z
		 * cache -- returns Z as well, useful during backpropagation 
		 */
		MyArray A = new MyArray(Z.getRows(), Z.getColumns());
		MyArray cache = new MyArray(Z.getRows(), Z.getColumns());
		for(int i=0;i<A.getRows();i++){
			for(int j=0;j<A.getColumns();j++){
				A.set(i, j, 1/(1+Math.exp(-Z.getNumber(i, j))));
				cache.set(i, j, Z.getNumber(i, j));
			}
		}
		TwoTuple<MyArray, MyArray> twoTuple = new TwoTuple<MyArray, MyArray>(A, cache);
		return twoTuple;
	}

	public static TwoTuple<MyArray, MyArray> relu(MyArray Z){
		/**
		 * Implement the RELU function.
		 * 
		 * Arguments:
		 * Z -- Output of the linear layer, of any shape
		 * 
		 * Returns:
		 * A -- Post-activation parameter, of the same shape as Z
		 * cache -- a dictionary containing "A" ; stored for computing the backward pass efficiently
		 */
		MyArray A = new MyArray(Z.getRows(), Z.getColumns());
		MyArray cache = new MyArray(Z.getRows(), Z.getColumns());
		for(int i=0;i<A.getRows();i++){
			for(int j=0;j<A.getColumns();j++){
				A.set(i, j, Math.max(0, Z.getNumber(i, j)));
				cache.set(i, j, Z.getNumber(i, j));
			}
		}
		TwoTuple<MyArray, MyArray> twoTuple = new TwoTuple<MyArray, MyArray>(A, cache);
		return twoTuple;
	}

	public static MyArray relu_backward(MyArray dA,MyArray cache){
		/**
		 * Implement the backward propagation for a single RELU unit.
		 * 
		 * Arguments:
		 * dA -- post-activation gradient, of any shape
		 * cache -- 'Z' where we store for computing backward propagation efficiently
		 * 
		 * Returns:
		 * dZ -- Gradient of the cost with respect to Z
		 */
		MyArray Z = cache;
		MyArray dZ = new MyArray(dA.getRows(), dA.getColumns());
		for(int i=0;i<dZ.getRows();i++){
			for(int j=0;j<dZ.getColumns();j++){
				if(Z.getNumber(i, j) <= 0){
					dZ.set(i, j, 0);
				}else{
					dZ.set(i, j, dA.getNumber(i, j));
				}
			}
		}
		return dZ;
	}

	public static MyArray sigmoid_backward(MyArray dA,MyArray cache){
		/**
		 * Implement the backward propagation for a single SIGMOID unit.
		 * 
		 * Arguments:
		 * dA -- post-activation gradient, of any shape
		 * cache -- 'Z' where we store for computing backward propagation efficiently
		 * 
		 * Returns:
		 * dZ -- Gradient of the cost with respect to Z
		 */
		
		MyArray Z = cache;
		MyArray s = new MyArray(Z.getRows(), Z.getColumns());
		MyArray dZ = new MyArray(Z.getRows(), Z.getColumns());
		for(int i=0;i<s.getRows();i++){
			for(int j=0;j<s.getColumns();j++){
				s.set(i, j, 1/(1+Math.exp(-Z.getNumber(i, j))));
				dZ.set(i, j, dA.getNumber(i, j)*s.getNumber(i, j)*(1-s.getNumber(i, j)));
			}
		}
		return dZ;
	}
}
