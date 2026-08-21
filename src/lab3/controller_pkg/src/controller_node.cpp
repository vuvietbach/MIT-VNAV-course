#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <mav_msgs/msg/actuators.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>

#define PI M_PI

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  PART 0 |  16.485 - Fall 2024  - Lab 3 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a geometric controller for a
//  simulated UAV, following the publication:
//
//  [1] Lee, Taeyoung, Melvin Leoky, N. Harris McClamroch. "Geometric tracking
//      control of a quadrotor UAV on SE (3)." Decision and Control (CDC),
//      49th IEEE Conference on. IEEE, 2010
//
//  We use variable names as close as possible to the conventions found in the
//  paper, however, we have slightly different conventions for the aerodynamic
//  coefficients of the propellers (refer to the lecture notes for these).
//  Additionally, watch out for the different conventions on reference frames
//  (see Lab 3 Handout for more details).
//
//  The include below is strongly suggested [but not mandatory if you have
//  better alternatives in mind :)]. Eigen is a C++ library for linear algebra
//  that will help you significantly with the implementation. Check the
//  quick reference page to learn the basics:
//
//  https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

#include <eigen3/Eigen/Dense>
typedef Eigen::Matrix<double, 4, 4> Matrix4d;

// If you choose to use Eigen, tf2 provides useful functions to convert tf2
// messages to eigen types and vice versa.
#include <tf2_eigen/tf2_eigen.hpp>

// FOR exit(1) FOR DEBUGGING
#include <cstdlib>

class ControllerNode : public rclcpp::Node {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  PART 1 |  Declare ROS callback handlers
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //
  // In this section, you need to declare:
  //   1. two subscribers (for the desired and current UAVStates)
  //   2. one publisher (for the propeller speeds)
  //   3. a timer for your main control loop
  //
  // ~~~~ begin solution
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr
      sub_desired_state_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_current_state_;
  rclcpp::Publisher<mav_msgs::msg::Actuators>::SharedPtr pub_rotor_speeds_;

  rclcpp::TimerBase::SharedPtr timer_;
  // ~~~~ end solution
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                                 end part 1
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Controller parameters
  double kx, kv, kr, komega; // controller gains - [1] eq (15), (16)

  // Physical constants (we will set them below)
  double m;            // mass of the UAV
  double g;            // gravity acceleration
  double d;            // distance from the center of propellers to the c.o.m.
  double cf,           // Propeller lift coefficient
      cd;              // Propeller drag coefficient
  Eigen::Matrix3d J;   // Inertia Matrix
  Eigen::Vector3d e3;  // [0,0,1]
  Eigen::Matrix4d F2W; // Wrench-rotor speeds map

  // Controller internals (you will have to set them below)
  // Current state
  Eigen::Vector3d x; // current position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d v; // current velocity of the UAV's c.o.m. in the world frame
  Eigen::Matrix3d R; // current orientation of the UAV
  Eigen::Vector3d
      omega; // current angular velocity of the UAV's c.o.m. in the *body* frame

  // Desired state
  Eigen::Vector3d xd; // desired position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d vd; // desired velocity of the UAV's c.o.m. in the world frame
  Eigen::Vector3d
      ad;      // desired acceleration of the UAV's c.o.m. in the world frame
  double yawd; // desired yaw angle

  int64_t hz; // frequency of the main control loop

  static Eigen::Vector3d Vee(const Eigen::Matrix3d &in) {
    Eigen::Vector3d out;
    out << in(2, 1), in(0, 2), in(1, 0);
    return out;
  }

  static double signed_sqrt(double val) {
    return val > 0 ? sqrt(val) : -sqrt(-val);
  }

public:
  ControllerNode() : Node("controller_node"), e3(0, 0, 1), F2W(4, 4), hz(1000) {
    // declare ROS parameters
    declare_parameter<double>("kx");
    declare_parameter<double>("kv");
    declare_parameter<double>("kr");
    declare_parameter<double>("komega");

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  INITIALIZE STATE VARIABLES (ADD THIS BLOCK)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x.setZero();
    v.setZero();
    omega.setZero();
    R.setIdentity();

    xd.setZero();
    vd.setZero();
    ad.setZero();
    yawd = 0.0;
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 2 |  Initialize ROS callback handlers
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //
    // In this section, you need to initialize your handlers from part 1.
    // Specifically:
    //  - bind controllerNode::onDesiredState() to the topic "desired_state"
    //  - bind controllerNode::onCurrentState() to the topic "current_state"
    //  - bind controllerNode::controlLoop() to the created timer, at frequency
    //    given by the "hz" variable
    //
    // Hints:
    //  - make sure you start your timer with reset()
    //
    // ~~~~ begin solution
    sub_desired_state_ =
        this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
            "desired_state", 10,
            std::bind(&ControllerNode::onDesiredState, this, std::placeholders::_1));

    sub_current_state_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "current_state", 10,
        std::bind(&ControllerNode::onCurrentState, this, std::placeholders::_1));

    pub_rotor_speeds_ = this->create_publisher<mav_msgs::msg::Actuators>(
        "rotor_speed_cmds", 10);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000 / hz),
        std::bind(&ControllerNode::controlLoop, this));
    // ~~~~ end solution
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //                                 end part 2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (!(get_parameter("kx", kx) && get_parameter("kv", kv) &&
          get_parameter("kr", kr) && get_parameter("komega", komega))) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to get controller gains from parameter server");
      exit(1);
    }

    RCLCPP_INFO(this->get_logger(),
                "Controller gains loaded: kx = %f, kv = %f, kr = %f, komega = %f",
                kx, kv, kr, komega);

    // Initialize constants
    m = 1.0;
    cd = 1e-5;
    cf = 1e-3;
    g = 9.81;
    d = 0.3;
    J << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    double a = cf * d / sqrt(2);
    F2W << cf, cf, cf, cf, a, a, -a, -a, -a, a, a, -a, cd, -cd, cd, -cd;
  }

  void onDesiredState(
      const trajectory_msgs::msg::MultiDOFJointTrajectoryPoint &des_state) {

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 3 | Objective: fill in xd, vd, ad, yawd
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //
    // 3.1 Get the desired position, velocity and acceleration from the in-
    //     coming ROS message and fill in the class member variables xd, vd
    //     and ad accordingly. You can ignore the angular acceleration.
    //
    // Hint: use "v << vx, vy, vz;" to fill in a vector with Eigen.
    //

    // 3.1 Get the desired position, velocity and acceleration
    xd << des_state.transforms[0].translation.x,
        des_state.transforms[0].translation.y,
        des_state.transforms[0].translation.z;
    vd << des_state.velocities[0].linear.x, des_state.velocities[0].linear.y,
        des_state.velocities[0].linear.z;
    ad << des_state.accelerations[0].linear.x,
        des_state.accelerations[0].linear.y,
        des_state.accelerations[0].linear.z;

    // 3.2 Extract the yaw component
    tf2::Quaternion q;
    tf2::fromMsg(des_state.transforms[0].rotation, q);
    yawd = tf2::getYaw(q);
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //                                 end part 3
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }

  void onCurrentState(const nav_msgs::msg::Odometry &cur_state) {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 4 | Objective: fill in x, v, R and omega
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //
    // Get the current position and velocity from the incoming ROS message and
    // fill in the class member variables x, v, R and omega accordingly.
    //
    //  CAVEAT: cur_state.twist.twist.angular is in the world frame, while omega
    //          needs to be in the body frame!
    //

    x << cur_state.pose.pose.position.x, cur_state.pose.pose.position.y,
        cur_state.pose.pose.position.z;

    Eigen::Quaterniond q(
        cur_state.pose.pose.orientation.w, cur_state.pose.pose.orientation.x,
        cur_state.pose.pose.orientation.y, cur_state.pose.pose.orientation.z);
    R = q.toRotationMatrix();

    v << cur_state.twist.twist.linear.x, cur_state.twist.twist.linear.y,
        cur_state.twist.twist.linear.z;
    

    Eigen::Vector3d omega_world;
    omega_world << cur_state.twist.twist.angular.x,
        cur_state.twist.twist.angular.y, cur_state.twist.twist.angular.z;
    omega = R.transpose() * omega_world;
  }

  void controlLoop() {
    Eigen::Vector3d ex, ev, er, eomega;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 5 | Objective: Implement the controller!
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //
    // 5.1 Compute position and velocity errors. Objective: fill in ex, ev.
    ex = x - xd;
    ev = v - vd;

    // 5.2 Compute the Rd matrix.
    Eigen::Vector3d F = -kx * ex - kv * ev + m * g * e3 + m * ad;
    Eigen::Vector3d b3d = F.normalized();

    Eigen::Vector3d b1c(cos(yawd), sin(yawd), 0);
    Eigen::Vector3d b2d = (b3d.cross(b1c)).normalized();
    Eigen::Vector3d b1d = b2d.cross(b3d);

    Eigen::Matrix3d Rd;
    Rd << b1d, b2d, b3d;

    // 5.3 Compute the orientation error (er) and the rotation-rate error
    er = 0.5 * Vee(Rd.transpose() * R - R.transpose() * Rd);
    eomega = omega; // ignoring omega_d as suggested

    // 5.4 Compute the desired wrench (force + torques)
    double f = F.dot(R.col(2));
    Eigen::Vector3d M = -kr * er - komega * eomega + omega.cross(J * omega);

    // 5.5 Recover the rotor speeds from the wrench
    Eigen::Vector4d wrench;
    wrench << f, M(0), M(1), M(2);
    Eigen::Vector4d rotor_speeds_squared = F2W.inverse() * wrench;

    // 5.6 Populate and publish the control message
    mav_msgs::msg::Actuators msg;
    msg.header.stamp = this->get_clock()->now();
    for (int i = 0; i < 4; i++) {
      msg.angular_velocities.push_back(signed_sqrt(rotor_speeds_squared(i)));
    }
    pub_rotor_speeds_->publish(msg);
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //           end part 5, congrats! Start tuning your gains (part 6)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv); // Initialize the ROS 2 system
  rclcpp::spin(std::make_shared<ControllerNode>()); // Spin the node so it
                                                    // processes callbacks
  rclcpp::shutdown(); // Shutdown the ROS 2 system when done
  return 0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  PART 6 [NOTE: save this for last] |  Tune your gains!
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
// Live the life of a control engineer! Tune these parameters for a fast
// and accurate controller.
//
// Modify the gains kx, kv, kr, komega in controller_pkg/config/params.yaml
// and re-run the controller.
//
// Can you get the drone to do stable flight?
//
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//  You made it! Congratulations! You are now a control engineer!
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
