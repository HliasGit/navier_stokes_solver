// Gmsh project created on Sun Sep 15 13:41:59 2024
SetFactory("OpenCASCADE");

// Define the rectangle
Rectangle(1) = {0.0, 0.0, 0, 2.2, 0.41, 0};

// Define the circle (cylinder)
Circle(5) = {0.2, 0.2, 0, 0.05, 0, 2*Pi};

// Create surface for the rectangle and circle
Curve Loop(1) = {1, 2, 3, 4};  // Rectangle boundary (ensure counterclockwise)
Plane Surface(1) = {1};

Curve Loop(2) = {5};  // Circle boundary
Plane Surface(2) = {2};  // This is the circular hole

// Subtract the circular surface from the rectangle
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; }

// Physical curves for boundary conditions
Physical Curve("inlet", 7) = {7};     // Inlet curve (check the correct curve ID for inlet)
Physical Curve("outlet", 8) = {8};    // Outlet curve (check the correct curve ID for outlet)
Physical Curve("wall", 6) = {9, 6};   // Wall curves (ensure proper orientation)
Physical Curve("cylinder", 10) = {5}; // Cylinder curve (outer boundary of the circle)

// Ensure consistent mesh density along curves
Transfinite Curve {9, 6} = 150 Using Progression 1;  // Wall curves
Transfinite Curve {5} = 300 Using Progression 1;     // Cylinder curve
Transfinite Curve {7, 8} = 100 Using Progression 1;  // Inlet and outlet curves

// Define the main surface (excluding the cylinder surface)
Physical Surface(11) = {1};  // The main domain surface (Rectangle minus cylinder)
//+
Physical Surface(11) -= {2};
//+
Physical Surface(11) -= {2};
//+
Physical Surface(11) -= {2};
//+
Recursive Delete {
  Surface{2}; 
}
//+
Transfinite Curve {9} = 200 Using Progression 1;
//+
Transfinite Curve {6} = 200 Using Progression 1;
//+
Transfinite Curve {5} = 300 Using Progression 1;
//+
Transfinite Curve {9, 9} = 300 Using Progression 1;
//+
Transfinite Curve {9, 6} = 150 Using Progression 1;
//+
Transfinite Curve {8, 7} = 100 Using Progression 1;
