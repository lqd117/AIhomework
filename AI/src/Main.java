import java.awt.Window;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Event;
import java.awt.FlowLayout;
import java.awt.RenderingHints.Key;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Toolkit;
import java.beans.Expression;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import java.lang.Math;
import java.math.MathContext;
import java.time.Year;

import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.math3.analysis.function.Multiply;

import MyTuple.ThreeTuple;
import MyTuple.TwoTuple;



public class Main {
	
	public static void main(String[] args){
		JFrame jFrame = new JFrame("神经网络");
		jFrame.setBackground(Color.WHITE);
		jFrame.setVisible(true);
		jFrame.setMinimumSize(new Dimension(1080, 720));
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		jFrame.setLocation(screenSize.width/2-1080/2, screenSize.height/2-720/2);
		jFrame.setResizable(false);
		jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JPanel topJPanel = new JPanel();
		
		JTextArea expressionName = new JTextArea("表达式: ");
		JTextField expression = new JTextField("x*x",10);
		
		JTextArea layerName = new JTextArea("层数和节点数: ");
		JTextField layer = new JTextField("1,5,4,1",10);
		
		JTextArea exampleName = new JTextArea("样本数: ");
		JTextField example = new JTextField("1000",10);
		
		JTextArea batchName = new JTextArea("迭代数: ");
		JTextField batch = new JTextField("1000",10);
		
		JTextArea learning_rate_name = new JTextArea("学习速率");
		JTextField learning_rate = new JTextField("0.1",10);
		
		
		JButton button = new JButton("确定");
		button.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent arg0) {
				// TODO Auto-generated method stub
				System.out.println(expression.getText());
				
			}
		});
		
		topJPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		topJPanel.setBackground(Color.WHITE);
		topJPanel.add(expressionName);topJPanel.add(expression);
		topJPanel.add(layerName);topJPanel.add(layer);
		topJPanel.add(exampleName);topJPanel.add(example);
		topJPanel.add(layerName);topJPanel.add(layer);
		topJPanel.add(learning_rate_name);topJPanel.add(learning_rate);
		topJPanel.add(button);
		
		
		jFrame.add(topJPanel);
		
		jFrame.pack();
		
	}
}



