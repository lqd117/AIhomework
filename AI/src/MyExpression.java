import java.util.HashMap;
import java.util.Map;

public class MyExpression {
	private String expression;
	private Map<String, Object> map = new HashMap<>();
	public void init(String string){
		this.expression = string;
		map.put("math", new MyMath());
	}
	public  double getYByX(double x){
		map.put("x", x);
		return (double) StringToExpression.invokeMethod(expression, map);
	}
}
