// Vehicle Routing Problem for Cold Chain Logistics (translated from Python to CPLEX OPL)

int n = 6; // number of nodes (1 depot + 5 stores)
int m = 3; // number of vehicles
range Nodes = 0..n-1;
range Customers = 1..n-1;
range Vehicles = 0..m-1;
range I = 0..n-1;
range J = 0..n-1;
range K = 0..m-1;

// Coordinates (x, y), demand and service time for each node
float x[Nodes] = [0, 6, 15, 12, 3, 18];    
float y[Nodes] = [0, 7, 14, 5, 13, 9];
float demand[Nodes] = [0, 620, 640, 780, 810, 600];
float service_time[Nodes] = [0, 0.13, 0.14, 0.17, 0.18, 0.13];

float v = 35;
float c_cd = 645000;
float c_vc = 3000;
float c_t = 30000;
float c_s = 45000;
float p = 15000;
float phi_1 = 0.05;
float phi_2 = 0.075;
float q_k = 2000;

// Declare distance and travel time matrices
float distance[Nodes][Nodes];
float travel_time[Nodes][Nodes];

// Decision variables
dvar boolean X[I][J][K];
dvar float+ f[I][J][K];

// Objective function (with linear approximation of exp)
minimize
  sum(j in Customers, k in Vehicles) c_cd * X[0][j][k] +
  sum(i in Nodes, j in Nodes : i != j, k in Vehicles)
    (c_vc * distance[i][j] * X[i][j][k] +
     X[i][j][k] * (c_t * travel_time[i][j] + c_s * service_time[j]) +
     X[i][j][k] * p * (
        demand[j] * phi_1 * travel_time[i][j] +
        maxl(0, f[i][j][k] - demand[j]) * phi_2 * service_time[j]
     ));

subject to {
  // Each customer is visited exactly once
  forall(j in Customers)
    sum(i in Nodes : i != j, k in Vehicles) X[i][j][k] == 1;

  // Vehicle capacity and depot constraints
  forall(k in Vehicles) {
    sum(i in Nodes, j in Customers : i != j) demand[j] * X[i][j][k] <= q_k;
    sum(j in Customers) X[0][j][k] == 1;
    sum(j in Customers) X[j][0][k] == 1;
    forall(h in Customers)
      sum(i in Nodes : i != h) X[i][h][k] == sum(j in Nodes : j != h) X[h][j][k];
  }

  // Flow constraints
  forall(i in Nodes, j in Nodes : i != j, k in Vehicles) {
    f[i][j][k] >= demand[j] * X[i][j][k];
    f[i][j][k] <= q_k * X[i][j][k];
  }

  forall(j in Customers, k in Vehicles)
    sum(i in Nodes : i != j) f[i][j][k] - sum(i in Nodes : i != j) f[j][i][k] == 
    demand[j] * sum(i in Nodes : i != j) X[i][j][k];
}

// Compute distance and travel_time matrices
execute INITIALIZE {
  for (var i in Nodes) {
    for (var j in Nodes) {
      var dx = x[i] - x[j];
      var dy = y[i] - y[j];
      distance[i][j] = (dx * dx + dy * dy)^0.5;
      travel_time[i][j] = distance[i][j] / v;
    }
  }
}

// Allow solving non-convex objective
execute {
  cplex.OptimalityTarget = 3;
}

// Use this block in OPL IDE to get outputs if needed
execute DISPLAY {
  writeln("Model ready for solving. Visual output and route printing should be handled separately.");
}
