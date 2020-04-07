#include <iostream>
#include <fstream>
#include <algorithm> //sort
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

//keeps track of the accuracy of each feature 
struct Accuracy 
{
    double correctly_predicted;
    double total;
};

struct Neighbor 
{
    int columnOfFeature;
    double distance;
};

//for sort function to use as a comparison 
bool compareLength(const Neighbor &a, const Neighbor &b)
{
    return a.distance < b.distance;
};

int chosen_Algorithm; //holds which search algorithm is chosen
double file_data_matrix[3000][100]; //holds the data index
int NumberOfFeatures; //number of features 
int NumberOfInstances; //number of instances 
int k_NN = 0;

//Asks for user input for which file to be tested. Then algorithm to implement and the
void user_input()
{
    string testFile;
    cout << "Welcome to Yash Kelkar's Feature Selection Algorithm \n";
    Check: cout << "Type in the name of the file to test: "; 
    cin >> testFile; //Selects test file 
    
    ifstream file;
    file.open(testFile.c_str()); //Opens file chosen
    if(!file)
    {
        cerr << "File does not exist. Enter a valid input file." << endl; //Outputs if file does not exist 
        goto Check;
    }
    
    double instanceFromDataFile; //Holds values from data file
    int temp_column_counter = 0; //temporarily hold which feature we are inputting
    int row_counter = 0; //checkes which row we are on 
    int features = 0; //which feature we are on 
    bool start_filling_matrix = false; //purpose is to check if we are checking the first column and makes sure we skip it  
    
    //input data from input file and stores training instance into input_matrix
    while (!file.eof()) 
    {
        file >> instanceFromDataFile; //Stores data from text file into variable instanceFromDataFile
        //checks which column and skips if it is the column identifying the class
        //this will be skipped after 
        if ((instanceFromDataFile == 1 || instanceFromDataFile == 2) && start_filling_matrix == true) // checks which class the input value is in 
        {
            row_counter++; //increments the row to check the next row;
            features = temp_column_counter;
            temp_column_counter = 0;
            
        }
        start_filling_matrix = true; //starts imputting into data matrix after first column only
        
        file_data_matrix[row_counter][temp_column_counter] = instanceFromDataFile;
        temp_column_counter++; //move on to next column
        
    }
    
    row_counter++; //doesn't count the first row when taking in input so increments one more time 
    file.close(); //closes input file 
    
    NumberOfInstances = row_counter; //set global variable to number of intances data matrix
    NumberOfFeatures = features; //set global variable to number of features in data matrix
    
    cout << "Type the number of the algorithm you want to run. \n";
    cout << "1) Forward Selection\n2) Backward Elimination\n3) Yash's Special Algorithm\n";
    
    cin >> chosen_Algorithm;
    
    cout << "This dataset has " << NumberOfFeatures-1 << " features (not including "; //subtract one because it includes the class column
    cout << "the class attribute), with " << NumberOfInstances << " instances." << endl;
    
}

double EuclideanDistance(vector<double> training_set, vector<double> InstancesToBeClassified)
{
    double distance = 0;
    for(unsigned int x = 0; x < training_set.size(); x++)
    {
        distance += pow(training_set.at(x)-InstancesToBeClassified.at(x), 2);
    }
    distance = sqrt(distance);
    
    return distance;
}

//calculates euclidean distance from new instance to all other instance for features selected 
//instead of returning the predicted class label it stores the euclidean distances 
void NN_classifier(vector <Neighbor>& total_matrix, vector <double>& training_data, vector <int> selected_features, int i) 
{
    double distance = 0;
    Neighbor value;
    //loop through all rows except the same one being tested 
    for (int c = 0; c < NumberOfInstances; c++)
    {
        if (i != c) //makes sure we are not checking the same instance
        {
            distance = 0;
            vector<double> InstancesToBeClassified;
            //create a temporary list that stores all the instances to be classified 
            for (unsigned int a = 0; a < selected_features.size(); a++)
            {
                InstancesToBeClassified.push_back(file_data_matrix[c][selected_features.at(a)]);
            }
            //calculation for Euclidean Distance (distance between specified row and all other rows)
            distance = EuclideanDistance(training_data, InstancesToBeClassified);
            
            //store all totals into a list 
            
            value.distance = distance;
            value.columnOfFeature = c; 
            
            total_matrix.push_back(value); //returns a vector with all euclidean distances from new intance to all instances
            
        }
    }
    return;
}

Accuracy leave_one_out_validation(vector<int> selected_features, int knn)
{
    int correctly_predicted, total_amount = 0;
    Neighbor class_predicted;
    
    //sets k nearest neigbor based on the number of features 
    // if (NumberOfFeatures >= 40)
    // {
    //     k_NN = NumberOfFeatures/2;
    // }
    // else if (NumberOfFeatures < 40 && NumberOfFeatures >= 10)
    // {
    //     k_NN = NumberOfFeatures/2;
    // }
    // else
    // {
    k_NN = 1;
    // }
    
    for (int i = 0; i < NumberOfInstances; i++)
    {
        vector<Neighbor> total_matrix; //an instance of neighbor struct which holds all nearest neighbors used to calculate accuracy
        vector<double>training_set; //holds training phase
        
        //create a list that stores all of data in the training phase at current row
        for (unsigned int a = 0; a < selected_features.size(); a++)
        {
            training_set.push_back(file_data_matrix[i][selected_features.at(a)]);
        }
        
        NN_classifier(total_matrix, training_set, selected_features, i);
        
        //sort the list of totals from greatest to least
        sort(total_matrix.begin(), total_matrix.end(), compareLength); //orders euclidean distances from lowest to highest 
        int count1 = 0;
        int count2 = 0;
        //used for k nearest neighbor so it only checks k closest neighbors
        for (int a = 0; a < k_NN; a++)
        {
            if (file_data_matrix[total_matrix.at(a).columnOfFeature][0] == 1)
            {
                count1++; //keeps track of how many class 1 from the total_matrix (nearest neighbors from least distance to greatest)
            }
            else if (file_data_matrix[total_matrix.at(a).columnOfFeature][0] == 2)
            {
                count2++; //keeps track of how many class 2 (nearest neighbors from least distance to greatest)
            }
        }
        //increase total_count to calulate percent accuracy
        if (count1 >= count2) //if more nearest neighbors are class 1 we predict the instance is class 1
        {
            if (file_data_matrix[i][0] == 1) //if the instance being checked is class 1
            {
                correctly_predicted++;
            }
        }
        else 
        {
            if (file_data_matrix[i][0] == 2) //if the instance being checked is class 1
            {
                correctly_predicted++;
            }
        }
        
        total_amount++; 
    }
    
    //get the count and total used for most accurate percentage
    Accuracy accuracy;
    accuracy.correctly_predicted = double(correctly_predicted);
    accuracy.total = double(total_amount);
    
    return accuracy;
}
   

void selected_features_display(vector<int> selected_features) //helper function for print_display
{
    //gets rid of extra code cuts back on size 
    for (unsigned int c = 0; c < selected_features.size()-1; c++)
    {
        cout << selected_features.at(c) << ",";
    }
    cout << selected_features.at(selected_features.size()-1);
}

void print_display(Accuracy P, vector<int> current_features, int display_type)
{
    double percentage = (P.correctly_predicted/P.total)*100;
    
    sort(current_features.begin(), current_features.end());
    
    //different display messages depending on the results 
    if (display_type == 1)
    {
        cout << "Using feature(s) {";
        for (unsigned int c = 0; c < current_features.size(); c++)
        {
            if (c != current_features.size()-1) //for multiple features
            {
                cout << current_features.at(c) << ",";
            }
            else //when the last feature is reached or a single feature is used
            {
                cout << current_features.at(c);
            }
        }
        cout << "} accuracy is " << fixed << setprecision(1) << percentage << "%" << endl;
    }
    else if (display_type == 2)
    {
        reprint: cout << endl << "Feature set {";
        selected_features_display(current_features);
        cout << "} was best, with accuracy of " << fixed << setprecision(1) << percentage << "%" << endl << endl;
    }
    else if (display_type == 3)
    {
        cout << endl << "(Warning, Accuracy has decreased! Continuing search in case of local maxima) ";
        goto reprint;
    }
    else if (display_type == 4)
    {
        cout << "Finished search!! The best feature subset is {";
        selected_features_display(current_features);
        cout << "}, which has accuracy of " << fixed << setprecision(1) << percentage << "%" << endl;
    }
}

void Search_Algorithm()
{
    cout << endl << "Beginning search." << endl << endl;
    
    vector<int> current_features; //keeps track of which current_features we are checking
    vector<int> closest_features;
    int value;
    
    Accuracy Total_Best;
    Total_Best.correctly_predicted = 0;
    Total_Best.total = 0;
    
    //Forward Selection
    if (chosen_Algorithm == 1)
    {
        while (int(current_features.size()) < NumberOfFeatures-1) //continues as long as the size of the vector is less that the number of features 
        {
            Accuracy Local_Best;
            double best_value = 0;
            
            //for all features
            for (int i = 1; i < NumberOfFeatures; i++)
            {
                bool checked = true;
                //check to make sure feature is not in current features list 
                for (unsigned int j = 0; j < current_features.size(); j++)
                {
                    if (i == current_features.at(j)) //skips already in current features
                    {
                        checked = false;
                    }
                }
                if (checked)
                {
                    vector<int>temp = current_features;
                    temp.push_back(i); //populates the vector with the number feature 
                    
                    //call to nearest neighbor
                    Accuracy P = leave_one_out_validation(temp, k_NN);
                    print_display(P, temp, 1);
                    temp.clear();
                    
                    //check for best feature to be added 
                    if (P.correctly_predicted/P.total > best_value)
                    {
                        best_value = P.correctly_predicted/P.total;
                        Local_Best.correctly_predicted = P.correctly_predicted;
                        Local_Best.total = P.total;
                        value = i; //feature or features with the best percentage
                    }
                }
            }
            
            //push best local feature on selected_features 
            current_features.push_back(value);
            
            //checks to see total best feture list for all cases
            if (Local_Best.correctly_predicted/Local_Best.total > Total_Best.correctly_predicted/Total_Best.total && Total_Best.total != 0)
            {
                print_display(Local_Best, current_features, 2);
                closest_features = current_features;
                Total_Best.correctly_predicted = Local_Best.correctly_predicted;
                Total_Best.total = Local_Best.total;
            }
            else if (Local_Best.correctly_predicted/Local_Best.total <= Total_Best.correctly_predicted/Total_Best.total && Total_Best.total != 0)
            {
                print_display(Local_Best, current_features, 3);
            }
            else if (Total_Best.total == 0)
            {
                print_display(Local_Best, current_features, 2);
                closest_features = current_features;
                Total_Best.correctly_predicted = Local_Best.correctly_predicted;
                Total_Best.total = Local_Best.total;
            }
        }
        print_display(Total_Best, closest_features, 4);
    }
    
    //Backwards Elimination
    else if(chosen_Algorithm == 2)
    {
        bool start = true;
        
        //fill up the feature list with every feature
        for (int i = 1; i < NumberOfFeatures; i++)
        {
            current_features.push_back(i);
        }
        
        //while there are some features loop
        while (int(current_features.size()) > 1) 
        {
            Accuracy Local_Best;
            double best_value = 0; //a filler variable for Local Best 
            
            //check if first iteration 
            if (start == true)
            {
                vector<int>temp = current_features;
                Accuracy P = leave_one_out_validation(temp, k_NN);
                print_display(P, temp, 1);
                if (P.correctly_predicted/P.total > best_value)
                {
                    best_value = P.correctly_predicted/P.total;
                    Local_Best.correctly_predicted = P.correctly_predicted;
                    Local_Best.total = P.total;
                }
            }
            else
            {
                //loop through the features backwards
                for (int i = NumberOfFeatures-1; i > 0; i--)
                {
                    bool checked = true;
                    //check to see if feature is on list
                    for (unsigned int j = 0; j < current_features.size(); j++)
                    {
                        if (i == current_features.at(j))
                        {
                            checked = false; //if already checked
                        }
                    }
                    if (!checked)
                    {
                        vector<int> temp = current_features;
                        temp.erase(std::remove(temp.begin(), temp.end(), i), temp.end());
                        //call nearest neighbor
                        Accuracy P = leave_one_out_validation(temp, k_NN);
                        print_display(P, temp, 1);
                        if (P.correctly_predicted/P.total > best_value)
                        {
                            best_value = P.correctly_predicted/P.total;
                            Local_Best.correctly_predicted = P.correctly_predicted;
                            Local_Best.total = P.total;
                            value = i;
                        }
                    }
                }
                
                //pop element
                current_features.erase(remove(current_features.begin(), current_features.end(), value), current_features.end());
            }
            
            start = false;
            //check for total best element
            if (Local_Best.correctly_predicted/Local_Best.total >= Total_Best.correctly_predicted/Total_Best.total && Total_Best.total != 0)
            {
                print_display(Local_Best, current_features, 2);
                closest_features = current_features;
                Total_Best.correctly_predicted = Local_Best.correctly_predicted;
                Total_Best.total = Local_Best.total;
            }
            else if (Local_Best.correctly_predicted/Local_Best.total < Total_Best.correctly_predicted/Total_Best.total && Total_Best.total != 0)
            {
                print_display(Local_Best, current_features, 3);
            }
            else if (Total_Best.total == 0)
            {
                print_display(Local_Best, current_features, 2);
                closest_features = current_features;
                Total_Best.correctly_predicted = Local_Best.correctly_predicted;
                Total_Best.total = Local_Best.total;
            }
        }
        print_display(Total_Best, closest_features, 4);
    }
}

int main()
{
    user_input();
    Search_Algorithm();
    
    return 0;
}