import java.util.Scanner;

class StudentMarks {

    public static void main(String[] args) {
        // Declare variables to store marks of three subjects
        int cs, ds, od;
        // Declare variables to store total, average, grade, and highest subject
        int total;
        double average;
        String grade, highest;
        // Create a Scanner object to read input from the user
        Scanner sc = new Scanner(System.in);
        // Prompt the user to enter marks of three subjects
        System.out.print("CS ");
        cs = sc.nextInt();
        System.out.print("DS ");
        ds = sc.nextInt();
        System.out.print("OD ");
        od = sc.nextInt();
        // Close the scanner
        sc.close();
        // Calculate the total marks
        total = cs + ds + od;
        // Calculate the average marks
        average = total / 3.0;
        // Determine the grade based on the grading criteria
        if (cs < 35 || ds < 35 || od < 35) {
            grade = "Failed";
        } else if (average >= 85) {
            grade = "First Class with Distinction";
        } else if (average >= 70) {
            grade = "First Class";
        } else if (average >= 60) {
            grade = "Second Class";
        } else {
            grade = "Third Class";
        }
        // Determine the subject with the highest marks
        if (cs >= ds && cs >= od) {
            highest = "CS";
        } else if (ds >= cs && ds >= od) {
            highest = "DS";
        } else {
            highest = "OD";
        }
        // Display the results
        System.out.println("Sub\tMarks\tStatus");
        System.out.println("CS\t" + cs + "\t" + (cs < 35 ? "F" : "P"));
        System.out.println("DS\t" + ds + "\t" + (ds < 35 ? "F" : "P"));
        System.out.println("OD\t" + od + "\t" + (od < 35 ? "F" : "P"));
        System.out.println("Total: " + total);
        System.out.println("Average Marks: " + average);
        System.out.println("Grade: " + grade);
        System.out.println("Subjects with the Highest Marks: " + highest);
    }
}
