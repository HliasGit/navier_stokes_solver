// Gmsh project created on Thu Mar 21 18:01:23 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2.2, 0, 0, 1.0};
//+
Point(3) = {2.2, 0.41, 0, 1.0};
//+
Point(4) = {0, 0.41, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Circle(5) = {0.2, 0.2, 0, 0.05, 0, 2*Pi};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Curve Loop(2) = {5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("borders", 6) = {3, 1, 5};
//+
Physical Curve("inlet", 7) = {4};
//+
Physical Curve("outlet", 8) = {2};
//+
Physical Surface("plane", 9) = {1};//+
//+

//+
Transfinite Curve {5} = 180 Using Progression 1;
//+
Transfinite Curve {1, 3} = 110 Using Progression 1;
//+
Transfinite Curve {4, 2} = 25 Using Progression 1;
