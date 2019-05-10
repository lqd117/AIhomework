import java.util.HashMap;
import java.util.Map;

public class MyExpression {
	private static String expression;
	private static Map<String, Object> map = new HashMap<>();
	public static void init(String string){
		MyExpression.expression = string;
		map.put("math", new MyMath());
	}
	public static double getYByX(double x){
		map.put("x", x);
		return (double) StringToExpression.invokeMethod(expression, map);
	}
}
