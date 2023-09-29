from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import mtimes
from casadi import jacobian
import casadi as ca
from fancy_plots import fancy_plots_2, fancy_plots_1
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy


from geometry_msgs.msg import TwistStamped
import math
from std_msgs.msg import Header

#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

# Global variables Odometry Drone
x_real = 0.0
y_real = 0.0
z_real = 2.5
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0

# Angular velocities
qx_real = 0.0005
qy_real = 0.0
qz_real = 0.0
qw_real = 1.0
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0

hdp_vision = [0,0,0,0,0,0]
axes = [0,0,0,0,0,0]

def visual_callback(msg):

    global hdp_vision 

    vx_visual = msg.twist.linear.x
    vy_visual = msg.twist.linear.y
    vz_visual = msg.twist.linear.z
    wx_visual = msg.twist.angular.x
    wy_visual = msg.twist.angular.y
    wz_visual = msg.twist.angular.z

    hdp_vision = [vx_visual, vy_visual, vz_visual, wx_visual, wy_visual, wz_visual]

def rc_callback(data):
    # Extraer los datos individuales del mensaje
    global axes
    axes_aux = data.axes
    psi = -np.pi / 2

    R = np.array([[np.cos(psi), -np.sin(psi), 0, 0, 0, 0],
                [np.sin(psi), np.cos(psi), 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
    
    axes = R@axes_aux

def odometry_call_back(odom_msg):
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    # Read desired linear velocities from node
    x_real = odom_msg.pose.pose.position.x 
    y_real = odom_msg.pose.pose.position.y
    z_real = odom_msg.pose.pose.position.z
    
    qx_real = odom_msg.pose.pose.orientation.x
    qy_real = odom_msg.pose.pose.orientation.y
    qz_real = odom_msg.pose.pose.orientation.z
    qw_real = odom_msg.pose.pose.orientation.w

    vx_real = odom_msg.twist.twist.linear.x
    vy_real = odom_msg.twist.twist.linear.y
    vz_real = odom_msg.twist.twist.linear.z

    wx_real = odom_msg.twist.twist.angular.x
    wy_real = odom_msg.twist.twist.angular.y
    wz_real = odom_msg.twist.twist.angular.z

    
    return None


def calc_M(chi, a, b, x):
    w = x[7]

    M = MX.zeros(4, 4)
    M[0,0] = chi[0]
    M[0,1] = 0
    M[0,2] = 0
    M[0,3] = b * chi[1]
    M[1,0] = 0
    M[1,1] = chi[2]
    M[1,2] = 0
    M[1,3] = a* chi[3]
    M[2,0] = 0
    M[2,1] = 0
    M[2,2] = chi[4]
    M[2,3] = 0
    M[3,0] = b*chi[5]
    M[3,1] = a* chi[6]
    M[3,2] = 0
    M[3,3] = chi[7]*(a**2+b**2) + chi[8]
    
    return M

def calc_C(chi, a, b, x):
    w = x[7]

    C = MX.zeros(4, 4)
    C[0,0] = chi[9]
    C[0,1] = w*chi[10]
    C[0,2] = 0
    C[0,3] = a * w * chi[11]
    C[1,0] = w*chi[12]
    C[1,1] = chi[13]
    C[1,2] = 0
    C[1,3] = b * w * chi[14]
    C[2,0] = 0
    C[2,1] = 0
    C[2,2] = chi[15]
    C[2,3] = 0
    C[3,0] = a *w* chi[16]
    C[3,1] = b * w * chi[17]
    C[3,2] = 0
    C[3,3] = chi[18]

    return C

def calc_G():
    G = MX.zeros(4, 1)
    G[0, 0] = 0
    G[1, 0] = 0
    G[2, 0] = 0
    G[3, 0] = 0

    return G


def calc_J(x, a, b):
    
    psi = x[3]

    RotZ = MX.zeros(4, 4)

    RotZ[0, 0] = cos(psi)
    RotZ[0, 1] = -sin(psi)
    RotZ[0, 2] = 0
    RotZ[0, 3] = -(a*sin(psi) + b*cos(psi))
    RotZ[1, 0] = sin(psi)
    RotZ[1, 1] = cos(psi)
    RotZ[1, 2] = 0 
    RotZ[1, 3] = (a*cos(psi) - b*sin(psi))
    RotZ[2, 0] = 0
    RotZ[2, 1] = 0
    RotZ[2, 2] = 1
    RotZ[2, 3] = 0
    RotZ[3, 0] = 0
    RotZ[3, 1] = 0
    RotZ[3, 2] = 0
    RotZ[3, 3] = 1

    J = RotZ

    return J

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    chi = [0.6756,    1.0000,    0.6344,    1.0000,    0.4080,    1.0000,    1.0000,    1.0000,    0.2953,    0.5941,   -0.8109,    1.0000,    0.3984,    0.7040,    1.0000,    0.9365,    1.0000, 1.0000,    0.9752]# Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    psi = MX.sym('psi')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    w = MX.sym('w')

    # General vector of the states
    x = vertcat(nx, ny, nz, psi, ul, um, un, w)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')
    s = MX.sym('s')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref,s)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    psi_p = MX.sym('psi_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p,ny_p,nz_p,psi_p,ul_p,um_p,un_p,w_p)

    # Rotational Matrix
    a = 0
    b = 0
    J = calc_J(x, a, b)
    M = calc_M(chi,a,b, x)
    C = calc_C(chi,a,b, x)
    G = calc_G()

    # Crear matriz A
    A_top = horzcat(MX.zeros(4, 4), J)
    A_bottom = horzcat(MX.zeros(4, 4), -mtimes(inv(M), C))
    A = vertcat(A_top, A_bottom)

    # Crear matriz B
    B_top = MX.zeros(4, 4)
    B_bottom = inv(M)
    B = vertcat(B_top, B_bottom)

    # Crear vector aux
    aux = vertcat(MX.zeros(4, 1), -mtimes(inv(M), G))

    f_expl = MX.zeros(8, 1)
    f_expl = A @ x + B @ u[0:4] + aux
    f_x = A@x + aux
    g_x = B

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    psi_d = MX.sym('psi_d')

    ul_d = MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')
    force = MX.sym('w_d')

    p = vertcat(nx_d, ny_d, nz_d, psi_d, ul_d, um_d, un_d, w_d, force)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system, f_x, g_x

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((8,))
    return aux_x

def create_ocp_solver_description(x0, N_horizon, t_horizon, ul_max, ul_min, um_max, um_min, un_max, un_min, w_max, w_min, n, az, aux, z_max) -> AcadosOcp:
    # Function to create the optimization solver
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Model of the system defined before
    model, f_system, f_x, g_x = f_system_model()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    # Control error defition
    error = ocp.p[0:8,0] - model.x

    # get value external foorce
    force_external = ocp.p[8,0]

    # Get gauss force field
    z_system = model.x[2]
    aux_z = (((z_system - z_max)/aux)**n)/az
    value = np.exp(-aux_z)

    # Control gains optimization
    Q_mat = MX.zeros(8, 8)
    Q_mat[4, 4] = 1.1
    Q_mat[5, 5] = 1.1
    Q_mat[6, 6] = 1.1*(value)
    Q_mat[7, 7] = 1.1

    # Control gains optimization
    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1.3*(1/ul_max)
    R_mat[1, 1] = 1.3*(1/um_max)
    R_mat[2, 2] = 1.3*(1/un_max)
    R_mat[3, 3] = 1.3*(1/w_max)

    # Lyapunov function
    V = (1/2)*error.T@Q_mat@error
    delta_v = jacobian(V, model.x)
    lyapuuunov_p = delta_v@f_x + delta_v@g_x@model.u[0:4]
    T = 10
    alpha = 0.1

    V_f = Function('V_f',[model.x, model.p], [V])
    V_p_f = Function('V_p_f',[model.x, model.p, model.u], [lyapuuunov_p])

    # Lyapunov as barrier function
    h = V_p_f(model.x, model.p, model.u) + (-0.5*V_f(model.x, model.p)) - model.u[4]
    barrier_l = (-T*np.log(-h/alpha))
    # Set initial values P vector

    ocp.cost.cost_type = "EXTERNAL"
    #ocp.model.cost_expr_ext_cost = error.T @ Q_mat @error + model.u.T @ R_mat @ model.u - (2)*value
    #ocp.model.cost_expr_ext_cost = error.T @ Q_mat @error + model.u[0:4].T @ R_mat @ model.u[0:4]
    #ocp.model.cost_expr_ext_cost = model.u[0:4].T @ R_mat @ model.u[0:4] + 0.5*barrier_l + 1*(model.u[4]**2)
    ocp.model.cost_expr_ext_cost = error.T @ Q_mat @error + model.u[0:4].T @ R_mat @ model.u[0:4] - (4)*value

    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Set constraints
    ocp.constraints.lbu = np.array([ul_min, um_min, un_min, w_min])
    ocp.constraints.ubu = np.array([ul_max, um_max, un_max, w_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = x0

    # Set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.tol = 1e-4

    # Set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp


def get_odometry_simple():
    # Function to get the odometry of the system
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real

    # Get quaternion
    quaternion = [qx_real, qy_real, qz_real, qw_real ]
    r_quat = R.from_quat(quaternion)

    # Euler Angles of the system
    euler =  r_quat.as_euler('zyx', degrees = False)
    phi = euler[2]
    theta = euler[1]
    psi = euler[0]

    # Complete Jacobian of the system
    J = np.zeros((3, 3))
    J[0, 0] =  np.cos(psi)*np.cos(theta)
    J[0, 1] = np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi)
    J[0, 2] = np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)
    J[1, 0] = np.cos(theta)*np.sin(psi)
    J[1, 1] = np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)
    J[1, 2] = np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)
    J[2, 0] =  -np.sin(theta)
    J[2, 1] =  np.cos(theta)*np.sin(phi)
    J[2, 2] =  np.cos(phi)*np.cos(theta) 
    J_inv = np.linalg.inv(J)
    v = np.dot(J_inv, [vx_real, vy_real, vz_real])

    # Velocities respect with the body frame
    ul_real = v[0]
    um_real = v[1]
    un_real = v[2]

    x_state = [x_real,y_real,z_real,psi,ul_real,um_real,un_real, wz_real]
    return x_state


def send_velocity_control(u, vel_pub, vel_msg):
    # Funtio to send the control commands to the system
    vel_msg.header.frame_id = "base_link"
    vel_msg.header.stamp = rospy.Time.now()
    vel_msg.twist.linear.x = u[0]
    vel_msg.twist.linear.y = u[1]
    vel_msg.twist.linear.z = u[2]
    vel_msg.twist.angular.x = 0
    vel_msg.twist.angular.y = 0
    vel_msg.twist.angular.z = u[3]
    # Publish control values
    vel_pub.publish(vel_msg)

def euler_to_quaternion(roll, pitch, yaw):
    # Euler to quaternion transformation
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]

def send_state_to_topic(state_vector):
    # Function to get the internal states of the system
    publisher = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    # Create an Odometry message
    odometry_msg = Odometry()
    quaternion = euler_to_quaternion(0, 0, state_vector[3])
    odometry_msg.header.frame_id = "odom"
    odometry_msg.header.stamp = rospy.Time.now()
    odometry_msg.pose.pose.position.x = state_vector[0]
    odometry_msg.pose.pose.position.y = state_vector[1]
    odometry_msg.pose.pose.position.z = state_vector[2]
    odometry_msg.pose.pose.orientation.x = quaternion[1]
    odometry_msg.pose.pose.orientation.y = quaternion[2]
    odometry_msg.pose.pose.orientation.z = quaternion[3]
    odometry_msg.pose.pose.orientation.w = quaternion[0]
    odometry_msg.twist.twist.linear.x = state_vector[4]
    odometry_msg.twist.twist.linear.y = state_vector[5]
    odometry_msg.twist.twist.linear.z = state_vector[6]
    odometry_msg.twist.twist.angular.x = 0
    odometry_msg.twist.twist.angular.y = 0
    odometry_msg.twist.twist.angular.z = state_vector[7]
    # Publish the message
    publisher.publish(odometry_msg)


def get_odometry_simple_sim():
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    quaternion = [qx_real, qy_real, qz_real, qw_real ]
    r_quat = R.from_quat(quaternion)
    euler =  r_quat.as_euler('zyx', degrees = False)
    psi = euler[0]
    x_state = [x_real,y_real,z_real,psi,vx_real, vy_real, vz_real, wz_real]
    return x_state

def force_field(x, zd_max, n, az, aux):
    # Function to evaluate the field respect to and object
    z = x[2]
    aux_z = (((z - zd_max)/aux)**n)/az
    value = np.exp(-aux_z)
    return value

def main(vel_pub, vel_msg):
    # Initial Values System
    # Simulation Time
    t_final = 60
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 10
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Obstacles gains
    n = 24
    az = 1000
    aux = 2
    z_max = 2

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)


    # Vector Initial conditions
    x = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)

    for k in range(0, 100):
        # Read Real data
        tic = time.time()
        x[:, 0] = get_odometry_simple()
        # Loop_rate.sleep()
        while (time.time() - tic <= t_s):
                None
        print("Init System")

    # Reference Signal of the system
    xref = np.zeros((12, t.shape[0]), dtype = np.double)

    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    campo = np.zeros((1, t.shape[0]+1-N_prediction), dtype = np.double)

    # Limits Control values
    ul_max = 5
    um_max = 5
    un_max = 5
    w_max = 5

    ul_min = -ul_max
    um_min = -um_max
    un_min = -un_max
    w_min = -w_max

    # Create Optimal problem
    model, f, f_x, g_x = f_system_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, ul_max, ul_min, um_max, um_min, un_max, un_min, w_max, w_min, n, az, aux, z_max)

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Ex = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)
    Ey = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)
    Ez = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    campo[:, 0] = force_field(x[:, 0], z_max, n, az, aux)

    # Simulation System
    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()
                 # Reference Signal of the system
        
        condicion = axes[5]
        if condicion  == -4545.0:
            xref[4,k:] = 1.575*axes[0]
            xref[5,k:] = 1.575*axes[1]
            xref[6,k:] = 1.575*axes[3]
            xref[7,k:] = 1.575*axes[2]

        elif condicion == -10000.0:        
            xref[4,k:] = 1.575*hdp_vision[0]
            xref[5,k:] = 1.575*hdp_vision[1]
            xref[6,k:] = 1.575*hdp_vision[2]
            xref[7,k:] = 1.575*hdp_vision[5]
        
        else:
            xref[4,k:] = 0
            xref[5,k:] = 0
            xref[6,k:] = 0
            xref[7,k:] = 0
        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])


        for j in range(N_prediction):
            acados_ocp_solver.set(j, "p", np.array([0.0, 0.0, 0.0, 0.0, xref[4,k], xref[5,k], xref[6,k], xref[7,k], campo[0, k]]))
            #acados_ocp_solver.set(j, "p", np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, campo[0, k]]))
            #acados_ocp_solver.set(j, "p", np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, campo[0, k]]))
        

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        aux_control = acados_ocp_solver.get(0, "u")
        u_control[:, k] = aux_control[0:4]
        
        
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)
        
        x[:, k+1] = get_odometry_simple()
        campo[:, k+1] = force_field(x[:, k+1], z_max, n, az, aux)
        
        delta_t[:, k] = toc_solver


        print("v_real:", " ".join("{:.2f}".format(value) for value in np.round(x[4:8, k], decimals=2)))
        
        
        rate.sleep() 
        toc = time.time() - tic 

    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1, ax11 = fancy_plots_1()
    states_x, = ax11.plot(t[0:x.shape[1]], x[0,:],
                    color='#BB5651', lw=2, ls="-")
    states_y, = ax11.plot(t[0:x.shape[1]], x[1,:],
                    color='#69BB51', lw=2, ls="-")
    states_z, = ax11.plot(t[0:x.shape[1]], x[2,:],
                    color='#5189BB', lw=2, ls="-")

    ax11.set_ylabel(r"$[states]$", rotation='vertical')
    ax11.set_xlabel(r"$[t]$", labelpad=5)
    ax11.legend([states_x, states_y, states_z],
            [r'$x$', r'$y$', r'$z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig1.savefig("states_xyz.png")
    fig1


    fig2, ax12 = fancy_plots_1()
    campo_p, = ax12.plot(t[0:x.shape[1]], campo[0,:],
                    color='#BB5651', lw=2, ls="-")

    ax12.set_ylabel(r"$[states]$", rotation='vertical')
    ax12.set_xlabel(r"$[t]$", labelpad=5)
    ax12.legend([campo_p],
            [r'$f$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    #fig2.savefig("states_angles.eps")
    fig2.savefig("force.png")
    fig2


    fig3, ax13 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax13.set_xlim((t[0], t[-1]))

    time_1, = ax13.plot(t[0:delta_t.shape[1]],delta_t[0,:],
                    color='#00429d', lw=2, ls="-")
    tsam1, = ax13.plot(t[0:t_sample.shape[1]],t_sample[0,:],
                    color='#9e4941', lw=2, ls="-.")

    ax13.set_ylabel(r"$[s]$", rotation='vertical')
    ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax13.legend([time_1,tsam1],
            [r'$t_{compute}$',r'$t_{sample}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        odometry_topic = "/dji_sdk/odometry"
        velocity_subscriber = rospy.Subscriber(odometry_topic, Odometry, odometry_call_back)


        vision_sub = rospy.Subscriber("/dji_sdk/visual_servoing/vel/drone", TwistStamped, visual_callback, queue_size=10)
        RC_sub = rospy.Subscriber("/dji_sdk/rc", Joy, rc_callback, queue_size=10)
        
        velocity_topic = "/m100/velocityControl"
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher(velocity_topic, TwistStamped, queue_size=10)

        main(velocity_publisher, velocity_message)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
