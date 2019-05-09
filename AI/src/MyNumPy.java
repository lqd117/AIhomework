import org.omg.CORBA.PUBLIC_MEMBER;

public class MyNumPy {
	public NumpyRandom random;
	public MyNumPy(){
		this.random = new NumpyRandom();
	}
	public MyArray zeros(int x,int y){
		MyArray array = new MyArray(x, y);
		for(int i=0;i<x;i++){
			for(int j=0;j<y;j++){
				array.set(i, j, 0.0);
			}
		}
		return array;
	}
	
	public MyArray log(MyArray x){
		MyArray result = new MyArray(x.getRows(), x.getColumns());
		for(int i=0;i<result.getRows();i++){
			for(int j=0;j<result.getColumns();j++){
				result.set(i, j, Math.log(x.getNumber(i, j)));
			}
		}
		return result;
	}
	
	public MyArray dot(MyArray x,MyArray y){
		MyArray array = new MyArray(x.getRows(), y.getColumns());
		for(int i=0;i<array.getRows();i++){
			for(int j=0;j<array.getColumns();j++){
				double sum=0;
				for(int k=0;k<x.getColumns();k++){
					sum += x.getNumber(i, k) * y.getNumber(k, j);
				}
				array.set(i, j, sum);
			}
		}
		return array;
	}
	
}
