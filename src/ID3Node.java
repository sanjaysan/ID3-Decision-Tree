import java.util.ArrayList;

public class ID3Node {

    public boolean isLeaf;
    public ID3DecisionTree.Attribute attribute;
    public String attributeLabel;
    public int nodeType;
    public int class1Data;
    public int class2Data;
    public double threshold;
    public ArrayList<ID3Node> children;
    public int level;
}
