import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeSet;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class ID3DecisionTree 
{
	public static class Attribute
    {
        public String attrName;
        public int attrType;
        public String[] attrVals;
    }

    public static class NumericAttribute
    {
        public double entropy;
        public double threshold;
    }
    
    private static String classLabels[];
    private static ArrayList<Attribute> attrList = null;
    private static ArrayList<HashMap<String, String>> instances = null;
    private static int noOfAttrs = 0;

    private static double calcLog2(double value)
    {
        if (value == 0.0) 
        {
            return 0.0;
        }
        double numerator = Math.log(value);
        double denominator = Math.log(2.0);
        return numerator / denominator;
    }

    private static double calcEntropy(double numerator, double denominator) 
    {
        if (numerator == 0 || denominator == 0) 
        {
            return 0;
        }
        return (numerator / denominator) * calcLog2(numerator / denominator);
    }
    
    private static ID3Node buildTree(ArrayList<HashMap<String, String>> instances, int level, int m)
    {
        int classCnt1 = 0;
        int classCnt2 = 0;

        for (int i = 0; i < instances.size(); i++)
        {
            if (instances.get(i).get("class_label").equalsIgnoreCase(classLabels[0]))
            {
                classCnt1++;
            }
            else if (instances.get(i).get("class_label").equalsIgnoreCase(classLabels[1]))
            {
                classCnt2++;
            }
        }
        
//        System.out.println("ClassCount1: " + classCnt1);
//        System.out.println("ClassCount2: " + classCnt2);
        
        double p1 = (double)classCnt1 / (double)instances.size();
        double p2 = (double)classCnt2 / (double)instances.size();
        double totalEntropy = -((p1 * calcLog2(p1)) + (p2 * calcLog2(p2)));

        if (totalEntropy == 0.0)
        {
            ID3Node ID3Node = new ID3Node();
            ID3Node.isLeaf = true;
            ID3Node.attribute = null;
            ID3Node.children = null;
            ID3Node.threshold = 0.0;
            ID3Node.nodeType = -1;
            ID3Node.level = level;
            ID3Node.class1Data = classCnt1;
            ID3Node.class2Data = classCnt2;
            
//            System.out.println("ClassCount1: " + classCnt1);
//            System.out.println("ClassCount2: " + classCnt2);
            if (classCnt1 >= classCnt2)
            {
                ID3Node.attributeLabel = classLabels[0];
            }
            else
            {
                ID3Node.attributeLabel = classLabels[1];
            }
            return ID3Node;
        }

        if (instances.size() < m || instances.size() == 0)
        {
            ID3Node ID3Node = new ID3Node();
            ID3Node.isLeaf = true;
            ID3Node.attribute = null;
            ID3Node.children = null;
            ID3Node.threshold = 0.0;
            ID3Node.nodeType = -1;
            ID3Node.level = level;
            ID3Node.class1Data = classCnt1;
            ID3Node.class2Data = classCnt2;
//            System.out.println("ClassCount1: " + classCnt1);
//            System.out.println("ClassCount2: " + classCnt2);
            if (classCnt1 >= classCnt2)
            {
                ID3Node.attributeLabel = classLabels[0];
            }
            else
            {
                ID3Node.attributeLabel = classLabels[1];
            }
            return ID3Node;
        }

        double[] attrEntropy = new double[noOfAttrs];
        for (int i = 0; i < noOfAttrs; i++)
        {
            if (attrList.get(i).attrType == 0)
            {
                //Nominal attribute
                attrEntropy[i] = calcEntropyForNominalAttribute(i, instances);

            }
            else if (attrList.get(i).attrType == 1)
            {
                //Continuous attribute
                attrEntropy[i] = calcEntropyForNumericAttribute(i, instances).entropy;
            }
        }

        int minEntropyIndex = 0;
        double minEntropy = attrEntropy[0];

        int k = 1;
        while(k < attrEntropy.length)
        {
            if (attrEntropy[k] < minEntropy)
            {
                minEntropy = attrEntropy[k];
                minEntropyIndex = k;
            }
            k++;
        }

        Attribute attr = attrList.get(minEntropyIndex);
        if (attr.attrType == 0)
        {
            //Nominal attribute
            ID3Node splitNode = new ID3Node();
            splitNode.threshold = -1;
            splitNode.attributeLabel = "";
            splitNode.isLeaf = false;
            splitNode.nodeType = 0;
            splitNode.attribute = attr;
            splitNode.level = level;
            splitNode.class1Data = classCnt1;
            splitNode.class2Data = classCnt2;
            splitNode.children = new ArrayList<>(attr.attrVals.length);
//            System.out.println("SplitAttrName: "+ splitNode.attribute.attrName + " ClassCount1: " + classCnt1);
//            System.out.println("SplitAttrName: "+ splitNode.attribute.attrName + " ClassCount2: " + classCnt2);
            for (int i = 0; i < attr.attrVals.length; i++)
            {
                ArrayList<HashMap<String, String>> data = new ArrayList<>();
                for (int j = 0 ;j < instances.size(); j++)
                {
                    if (instances.get(j).get(attr.attrName).equalsIgnoreCase(attr.attrVals[i]))
                    {
                        data.add(instances.get(j));
                    }
                }
                splitNode.children.add(buildTree(data, level+1, m));
            }
            return splitNode;
        }
        else
        {
            //Continuous attribute
            int attributePos = -1;
            ID3Node splitNode = new ID3Node();
            int i = 0;
            while (i < attrList.size())
            {
                if (attrList.get(i).attrName.equalsIgnoreCase(attr.attrName))
                {
                    attributePos = i;
                    break;
                }
                i++;
            }
            splitNode.threshold = calcEntropyForNumericAttribute(attributePos, instances).threshold;
            splitNode.attributeLabel = "";
            splitNode.isLeaf = false;
            splitNode.nodeType = 0;
            splitNode.attribute = attr;
            splitNode.level = level;
            splitNode.class1Data = classCnt1;
            splitNode.class2Data = classCnt2;
            splitNode.children = new ArrayList<>(2);
//            System.out.println("SplitAttrName: "+ splitNode.attribute.attrName + " ClassCount1: " + classCnt1);
//            System.out.println("SplitAttrName: "+ splitNode.attribute.attrName + " ClassCount2: " + classCnt2);
            ArrayList<HashMap<String, String>> lessThanAttrValue = new ArrayList<>();
            ArrayList<HashMap<String, String>> greaterThanAttrValue = new ArrayList<>();

            for (int j = 0 ;j < instances.size(); j++) 
            {
                if (Double.parseDouble(instances.get(j).get(attr.attrName)) <= splitNode.threshold) 
                {
                    lessThanAttrValue.add(instances.get(j));
                } 
                else 
                {
                    greaterThanAttrValue.add(instances.get(j));
                }
            }
            splitNode.children.add(buildTree(lessThanAttrValue, level+1, m));
            splitNode.children.add(buildTree(greaterThanAttrValue, level+1, m));
            return splitNode;
        }
    }


    private static double calcEntropyForNominalAttribute(int attributePosition, ArrayList<HashMap<String, String>> instances)
    {
        double entropy = 0.0;

        int labelAttrCount = attrList.get(attributePosition).attrVals.length;
        String[] attributeValues = attrList.get(attributePosition).attrVals;
        int i = 0;
        while (i < labelAttrCount) 
        {
            int firstClassLabelCount = 0;
            int secondClassLabelCount = 0;
            int totalLabelCnt = 0;
            for (int j = 0; j < instances.size(); j++) 
            {
                if (instances.get(j).get(attrList.get(attributePosition).attrName).equalsIgnoreCase(attributeValues[i])) 
                {
                    if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[0]))
                    {
                        firstClassLabelCount++;
                    } 
                    else if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[1]))
                    {
                        secondClassLabelCount++;
                    }
                    totalLabelCnt++;
                }
            }
            double subEntropy = calcEntropy(firstClassLabelCount, totalLabelCnt) +
                            calcEntropy(secondClassLabelCount, totalLabelCnt);
            entropy += ((double) totalLabelCnt / instances.size()) * subEntropy;
            i++;
        }
        return -entropy;

    }

    private static NumericAttribute calcEntropyForNumericAttribute(int attributePosition, ArrayList<HashMap<String, String>> instances){
        double[] attrVals = new double[instances.size()];

        for (int i = 0; i < instances.size(); i++)
        {
            attrVals[i] = Double.parseDouble(instances.get(i).get(attrList.get(attributePosition).attrName));
        }
        Arrays.sort(attrVals);
        TreeSet<Double> distinctValues = new TreeSet<Double>();
        for (int i = 0; i < attrVals.length; i++)
        {
            distinctValues.add(attrVals[i]);
        }

        int k = 0;
        for (double d : distinctValues)
        {
            attrVals[k++] = d;
        }
        double[] potentialCandSplits = new double[distinctValues.size()];
        for (int i = 0; i < distinctValues.size()-1; i++)
        {
            potentialCandSplits[i] = (attrVals[i] + attrVals[i+1]) / 2.0;
        }
        attrVals = new double[instances.size()];

        for (int i = 0; i < instances.size(); i++)
        {
            attrVals[i] = Double.parseDouble(instances.get(i).get(attrList.get(attributePosition).attrName));
        }

        double entropy[] = new double[potentialCandSplits.length];


        for (int i = 0; i < potentialCandSplits.length; i++)
        {
            int lessThanClass1LabelCnt = 0;
            int GreaterThanClass1LabelCnt = 0;
            int lessThanClass2LabelCnt = 0;
            int GreaterThanClass2LabelCnt = 0;
            int lessThanCount = 0;
            int greaterThanCount = 0;
            for (int j = 0; j < instances.size(); j++)
            {
                if (attrVals[j] <= potentialCandSplits[i])
                {
                    lessThanCount++;
                    if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[0]))
                    {
                        lessThanClass1LabelCnt++;
                    }
                    else if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[1]))
                    {
                        lessThanClass2LabelCnt++;
                    }
                }
                else
                {
                    greaterThanCount++;
                    if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[0]))
                    {
                        GreaterThanClass1LabelCnt++;
                    }
                    else if (instances.get(j).get("class_label").equalsIgnoreCase(classLabels[1]))
                    {
                        GreaterThanClass2LabelCnt++;
                    }
                }
            }

            double p1 = 0.0, p2 = 0.0, ppos1 = 0.0;
            double ppos2 = 0.0, pneg1 = 0.0, pneg2 = 0.0;
            if (lessThanCount != 0) 
            {
                p1 = (double) lessThanCount / (double) instances.size();
                if (lessThanClass1LabelCnt != 0) 
                {
                    ppos1 = (double) lessThanClass1LabelCnt / (double) lessThanCount;
                } 
                else 
                {
                    ppos1 = 0.0;
                }

                if (lessThanClass2LabelCnt != 0) 
                {
                    ppos2 = (double) lessThanClass2LabelCnt / (double) lessThanCount;
                } 
                else 
                {
                    ppos2 = 0.0;
                }
            } 
            else 
            {
                p1 = 0.0;
                ppos1 = 0.0;
                ppos2 = 0.0;
            }

            if (greaterThanCount != 0) 
            {
                p2 = (double) greaterThanCount / (double) instances.size();
                if (GreaterThanClass1LabelCnt != 0) 
                {
                    pneg1 = (double) GreaterThanClass1LabelCnt / (double) greaterThanCount;
                } 
                else 
                {
                    pneg1 = 0.0;
                }

                if (GreaterThanClass2LabelCnt != 0) 
                {
                    pneg2 = (double) GreaterThanClass2LabelCnt / (double) greaterThanCount;
                } 
                else 
                {
                    pneg2 = 0.0;
                }
            } 
            else 
            {
                p2 = 0.0;
                pneg1 = 0.0;
                pneg2 = 0.0;
            }

            double log2ppos1 = 0.0, log2ppos2 = 0.0, log2pneg1 = 0.0, log2pneg2 = 0.0;
            if (ppos1 == 0.0) 
            {
                log2ppos1 = 0.0;
            } 
            
            else 
            {
                log2ppos1 = calcLog2(ppos1);
            }
            
            if (ppos2 == 0.0) 
            {
                log2ppos2 = 0.0;
            } 
            else
            {
                log2ppos2 = calcLog2(ppos2);
            }
            
            if (pneg1 == 0.0) 
            {
                log2pneg1 = 0.0;
            } 
            else 
            {
                log2pneg1 = calcLog2(pneg1);
            }
            
            if (pneg2 == 0.0) 
            {
                log2pneg2 = 0.0;
            } 
            else 
            {
                log2pneg2 = calcLog2(pneg2);
            }
            entropy[i] = -((p1 * ((ppos1 * log2ppos1) + (ppos2 * log2ppos2))) + (p2 * ((pneg1 * log2pneg1) + (pneg2 * log2pneg2))));
        }
        double minEntropy = entropy[0];
        int minEntropyIndex = 0;

        for (int i = 1; i < entropy.length; i++)
        {
            if (entropy[i] < minEntropy)
            {
                minEntropy = entropy[i];
                minEntropyIndex = i;
            }
        }

        NumericAttribute NumericAttribute = new NumericAttribute();
        NumericAttribute.entropy = minEntropy;
        NumericAttribute.threshold = potentialCandSplits[minEntropyIndex];
        return NumericAttribute;
    }

    private static String formatNumber(int decimals, double number) 
    {
        StringBuilder sb = new StringBuilder(decimals + 2);
        sb.append("0.");
        for (int i = 0; i < decimals; i++) 
        {
            sb.append("0");
        }
        return new DecimalFormat(sb.toString()).format(number);
    }

    private static void printDecisionTree(ID3Node node)
    {
        int nodeLevel = node.level;

        if (node.children != null) 
        {
            for (int j = 0; j < node.children.size(); j++) 
            {
            	int class1 = 0;
                int class2 = 0;
                
                if (node.level > 0) 
                {
                    for (int i = 0; i < nodeLevel; i++) 
                    {
                        System.out.print("|");
                        System.out.print("\t");
                    }
                }
                class1 = node.class1Data;
                class2 = node.class2Data; 
                System.out.print(node.attribute.attrName);
                if (node.attribute.attrType == 0) 
                {
                    System.out.print(" = ");
                    System.out.print(node.attribute.attrVals[j] + " [" + node.children.get(j).class1Data + " " + node.children.get(j).class2Data + "]");
                } 
                else if (node.attribute.attrType == 1) 
                {
                    if (node.children.indexOf(node.children.get(j)) == 0) 
                    {
                        System.out.print(" <= ");
                    } 
                    else if (node.children.indexOf(node.children.get(j)) == 1) 
                    {
                        System.out.print(" > ");
                    }
                    System.out.print(formatNumber(6, node.threshold) + " [" + node.children.get(j).class1Data + " " + node.children.get(j).class2Data + "]");
                }

                if (node.children.get(j).isLeaf) 
                {
                    System.out.print(": ");
                    if (node.children.get(j).attributeLabel == null) 
                    {
                        System.out.println(classLabels[0]);
                    } 
                    else 
                    {
                        System.out.println(node.children.get(j).attributeLabel);
                    }
                } 
                else 
                {
                    System.out.println();
                    printDecisionTree(node.children.get(j));
                }
            }
        }
    }
    
    private static String predictClass(ID3Node node, HashMap<String, String> data)
    {
        Attribute attribute = node.attribute;
        if (node.isLeaf) 
        {
            return node.attributeLabel;
        } 
        else 
        {
            if (attribute.attrType == 1) 
            {
                double splitValue = node.threshold;
                if (Double.parseDouble(data.get(attribute.attrName)) <= splitValue) 
                {
                    return (predictClass(node.children.get(0), data));
                } 
                else if (Double.parseDouble(data.get(attribute.attrName)) > splitValue) 
                {
                    return (predictClass(node.children.get(1), data));
                }
            } 
            else if (attribute.attrType == 0) 
            {
                String instanceAttributeValue = data.get(attribute.attrName);
                int attributeIndex = -1;
                for (int l = 0; l < attribute.attrVals.length; l++) 
                {
                    if (instanceAttributeValue.equalsIgnoreCase(attribute.attrVals[l])) 
                    {
                        attributeIndex = l;
                        break;
                    }
                }
                return (predictClass(node.children.get(attributeIndex), data));
            }
        }
        return "";
    }

    public static void main(String args[]){
        String arffTrainFile = args[0];
        String arffTestFile = args[1];
        int m = Integer.parseInt(args[2]);
    	
//    	String arffTrainFile = "/home/sanjay/MLProjects/DTree/dt-learn/Datset/heart_train.arff";
//        String arffTestFile = "/home/sanjay/MLProjects/DTree/dt-learn/Datset/heart_test.arff";
//        int m = 20;

        ID3Node root = trainDTree(arffTrainFile, m);
        validateDTree(arffTestFile, root);

    }

    private static void validateDTree(String arffTestFile, ID3Node node){
        ArffLoader arffLoader = new ArffLoader();
        File inputFile = new File(arffTestFile);
        try 
        {
            arffLoader.setFile(inputFile);
            Instances instance = arffLoader.getDataSet();

            HashMap<String, String> dataMap = new HashMap<String, String>();
            ArrayList<HashMap<String, String>> testData = new ArrayList<>(instance.numInstances());
            for (int k = 0; k < instance.numInstances(); k++)
            {
                dataMap = new HashMap<>();
                for (int l = 0; l < instance.numAttributes(); l++)
                {
                    if (l == instance.numAttributes() - 1)
                    {
                        dataMap.put("class_label", instance.instance(k).stringValue(l));
                    }
                    else 
                    {
                        if (attrList.get(l).attrType == 0) 
                        {
                            dataMap.put(attrList.get(l).attrName, instance.instance(k).stringValue(l));
                        } 
                        else if (attrList.get(l).attrType == 1) 
                        {
                            dataMap.put(attrList.get(l).attrName, String.valueOf(instance.instance(k).value(l)));
                        }
                    }
                }
                testData.add(dataMap);
            }

            int numCorrectlyClassified = 0;
            System.out.println("<Predictions for the Test Set Instances>");

            for (int i = 0; i < testData.size(); i++) 
            {
                String predictedClassLabel = "";
                String actualClassLabel = testData.get(i).get("class_label");
                System.out.print(String.format("%3d", i+1) + ": Actual: " + actualClassLabel + " Predicted: ");
                predictedClassLabel = predictClass(node, testData.get(i));
                if (predictedClassLabel.equalsIgnoreCase(actualClassLabel)) 
                {
                    numCorrectlyClassified++;
                }
                System.out.println(predictedClassLabel);
            }
            System.out.println("Number of correctly classified: " + numCorrectlyClassified +
                    " Total number of test instances: " + testData.size());
        }
        catch (IOException e)
        {
            System.out.println(e.getMessage());
        }
    }

    private static ID3Node trainDTree(String arffTrainFile, int m)
    {
        ArffLoader arffLoader = new ArffLoader();
        File inputFile = new File(arffTrainFile);
        try 
        {
            arffLoader.setFile(inputFile);
            Instances instance = arffLoader.getDataSet();
            classLabels = new String[2];
            classLabels[0] = instance.attribute(instance.numAttributes() - 1).value(0);
            classLabels[1] = instance.attribute(instance.numAttributes() - 1).value(1);

            attrList = new ArrayList<Attribute>(instance.numInstances() - 1);

            for (int i = 0; i < instance.numAttributes() - 1; i++)
            {
                Attribute attribute = new Attribute();
                attribute.attrName = instance.attribute(i).name();
                if (instance.attribute(i).isNominal()) 
                {
                    attribute.attrType = 0; // Nominal attribute
                    attribute.attrVals = new String[instance.attribute(i).numValues()];
                    for (int j = 0; j < instance.attribute(i).numValues(); j++)
                    {
                        attribute.attrVals[j] = instance.attribute(i).value(j);
                    }
                }
                else if (instance.attribute(i).isNumeric()) 
                {
                    attribute.attrType = 1; //Numeric attribute
                }
                attrList.add(attribute);
            }
            noOfAttrs = instance.numAttributes() - 1;

            HashMap<String, String> dataMap = new HashMap<String, String>();
            instances = new ArrayList<>(instance.numInstances());
            for (int k = 0; k < instance.numInstances(); k++)
            {
                dataMap = new HashMap<>();
                for (int l = 0; l < instance.numAttributes(); l++)
                {
                    if (l == instance.numAttributes() - 1)
                    {
                        dataMap.put("class_label", instance.instance(k).stringValue(l));
                    }
                    else 
                    {
                        if (attrList.get(l).attrType == 0)
                        {
                            dataMap.put(attrList.get(l).attrName, instance.instance(k).stringValue(l));
                        } 
                        else if (attrList.get(l).attrType == 1) 
                        {
                            dataMap.put(attrList.get(l).attrName, String.valueOf(instance.instance(k).value(l)));
                        }
                    }
                }
                instances.add(dataMap);
            }
            int rootLevel = 0;
            ID3Node root = buildTree(instances,rootLevel, m);
            printDecisionTree(root);
            return root;
        }
        catch (IOException e)
        {
            System.out.println(e.getMessage());
        }
        return null;
    }
}
