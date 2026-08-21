#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <cmath>

#include <mav_trajectory_generation/polynomial_optimization_linear.h>
#include <mav_trajectory_generation/trajectory.h>

#include <eigen3/Eigen/Dense>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class WaypointFollower : public rclcpp::Node {

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr currentStateSub;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr poseArraySub;

  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::
      SharedPtr desiredStatePub;

  // Current state
  Eigen::Vector3d x; // current position of the UAV's c.o.m. in the world frame

  rclcpp::TimerBase::SharedPtr desiredStateTimer;

  rclcpp::Time trajectoryStartTime;
  mav_trajectory_generation::Trajectory trajectory;
  mav_trajectory_generation::Trajectory yaw_trajectory;

  void onCurrentState(nav_msgs::msg::Odometry const &cur_state) {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 1.1 |  16.485 - Fall 2024  - Lab 4 coding assignment (5 pts)
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //
    //  Populate the variable x, which encodes the current world position of the
    //  UAV
    // ~~~~ begin solution
    x.x() = cur_state.pose.pose.position.x;
    x.y() = cur_state.pose.pose.position.y;
    x.z() = cur_state.pose.pose.position.z;
    // ~~~~ end solution
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //                                 end part 1.1
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }

  void
  generateOptimizedTrajectory(geometry_msgs::msg::PoseArray const &poseArray) {
    if (poseArray.poses.size() < 1) {
      RCLCPP_ERROR(get_logger(),
                   "Must have at least one pose to generate trajectory!");
      trajectory.clear();
      yaw_trajectory.clear();
      return;
    }

    if (!trajectory.empty())
      return;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 1.2 |  16.485 - Fall 2024  - Lab 4 coding assignment (35 pts)
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //
    //  We are using the mav_trajectory_generation library
    //  (https://github.com/ethz-asl/mav_trajectory_generation) to perform
    //  trajectory optimization given the waypoints (based on the position and
    //  orientation of the gates on the race course).
    //  We will be finding the trajectory for the position and the trajectory
    //  for the yaw in a decoupled manner.
    //  In this section:
    //  1. Fill in the correct number for D, the dimension we should apply to
    //  the solver to find the positional trajectory
    //  2. Correctly populate the Vertex::Vector structure below (vertices,
    //  yaw_vertices) using the position of the waypoints and the yaw of the
    //  waypoints respectively
    //
    //  Hints:
    //  1. Use vertex.addConstraint(POSITION, position) where position is of
    //  type Eigen::Vector3d to enforce a waypoint position.
    //  2. Use vertex.addConstraint(ORIENTATION, yaw) where yaw is a double
    //  to enforce a waypoint yaw.
    //  3. Remember angle wraps around 2 pi. Be careful!
    //  4. For the ending waypoint for position use .makeStartOrEnd as seen with
    //  the starting waypoint instead of .addConstraint as you would do for the
    //  other waypoints.
    //
    // ~~~~ begin solution
    // for access to SNAP
    using namespace mav_trajectory_generation::derivative_order;

    const int D = 3; // dimension of each vertex in the trajectory
    mav_trajectory_generation::Vertex::Vector vertices;
    mav_trajectory_generation::Vertex::Vector yaw_vertices;

    mav_trajectory_generation::Vertex start_position(D);
    start_position.makeStartOrEnd(x, SNAP);
    vertices.push_back(start_position);

    mav_trajectory_generation::Vertex start_yaw(1);
    start_yaw.addConstraint(ORIENTATION, 0.0);
    yaw_vertices.push_back(start_yaw);

    double last_yaw = 0.0;

    for (size_t i = 0; i < poseArray.poses.size(); ++i) {
      Eigen::Vector3d pos_eigen(
        poseArray.poses[i].position.x,
        poseArray.poses[i].position.y,
        poseArray.poses[i].position.z
      );

      mav_trajectory_generation::Vertex pos_vertex(D);
      if (i == poseArray.poses.size() - 1) {
        pos_vertex.makeStartOrEnd(pos_eigen, SNAP);
      } else {
        pos_vertex.addConstraint(POSITION, pos_eigen);
      }
      vertices.push_back(pos_vertex);

      const auto &q = poseArray.poses[i].orientation;
      double current_yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));

      // Unwrapping
      while (current_yaw - last_yaw > M_PI) current_yaw -= 2.0 * M_PI;
      while (current_yaw - last_yaw < -M_PI) current_yaw += 2.0 * M_PI;

      mav_trajectory_generation::Vertex yaw_vertex(1);
      yaw_vertex.addConstraint(ORIENTATION, current_yaw);
      yaw_vertices.push_back(yaw_vertex);

      last_yaw = current_yaw;
    }
    // ~~~~ end solution
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //                                 end part 1.2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ============================================================
    // Estimate the time to complete each segment of the trajectory
    // ============================================================

    std::vector<double> segment_times;
    const double v_max = 15.0;
    const double a_max = 10.0;
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);
    for (size_t i = 0; i < segment_times.size(); ++i) {
      segment_times[i] *= 0.6;
    }

    // =====================================================
    // Solve for the optimized trajectory (linear optimizer)
    // =====================================================
    // Position
    const int N = 10;
    mav_trajectory_generation::PolynomialOptimization<N> opt(D);
    opt.setupFromVertices(vertices, segment_times, SNAP);
    opt.solveLinear();

    // Yaw
    mav_trajectory_generation::PolynomialOptimization<N> yaw_opt(1);
    yaw_opt.setupFromVertices(yaw_vertices, segment_times, SNAP);
    yaw_opt.solveLinear();

    // ============================
    // Get the optimized trajectory
    // ============================
    mav_trajectory_generation::Segment::Vector segments;
    //        opt.getSegments(&segments); // Unnecessary?
    opt.getTrajectory(&trajectory);
    yaw_opt.getTrajectory(&yaw_trajectory);
    trajectoryStartTime = now();

    RCLCPP_INFO(get_logger(),
                "Generated optimizes trajectory from %zu waypoints",
                vertices.size());
  }

  void publishDesiredState() {
    if (trajectory.empty())
      return;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 1.3 |  16.485 - Fall 2024  - Lab 4 coding assignment (15 pts)
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //
    //  Finally we get to send commands to our controller! First fill in
    //  properly the value for 'nex_point.time_from_start' and 'sampling_time'
    //  (hint: not 0) and after extracting the state information from our
    //  optimized trajectory, finish populating next_point.
    //
    // ~~~~ begin solution
    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint next_point;

    // 1. Calculate time from start
    rclcpp::Duration time_from_start = now() - trajectoryStartTime;
    next_point.time_from_start = time_from_start;

    double sampling_time = time_from_start.seconds();

    // 2. Clamp to max time
    if (sampling_time > trajectory.getMaxTime()) {
      sampling_time = trajectory.getMaxTime();
    }

    // Evaluate trajectory
    using namespace mav_trajectory_generation::derivative_order;
    Eigen::Vector3d des_position = trajectory.evaluate(sampling_time, POSITION);
    Eigen::Vector3d des_velocity = trajectory.evaluate(sampling_time, VELOCITY);
    Eigen::Vector3d des_accel = trajectory.evaluate(sampling_time, ACCELERATION);
    Eigen::VectorXd des_orientation = yaw_trajectory.evaluate(sampling_time, ORIENTATION);

    // Populate next_point
    geometry_msgs::msg::Transform transform;
    transform.translation.x = des_position.x();
    transform.translation.y = des_position.y();
    transform.translation.z = des_position.z();
    
    double half_yaw = des_orientation(0) * 0.5;
    transform.rotation.x = 0.0;
    transform.rotation.y = 0.0;
    transform.rotation.z = sin(half_yaw);
    transform.rotation.w = cos(half_yaw);
    next_point.transforms.push_back(transform);

    geometry_msgs::msg::Twist velocity;
    velocity.linear.x = des_velocity.x();
    velocity.linear.y = des_velocity.y();
    velocity.linear.z = des_velocity.z();
    velocity.angular.x = 0.0;
    velocity.angular.y = 0.0;
    velocity.angular.z = 0.0;
    next_point.velocities.push_back(velocity);

    geometry_msgs::msg::Twist accel;
    accel.linear.x = des_accel.x();
    accel.linear.y = des_accel.y();
    accel.linear.z = des_accel.z();
    accel.angular.x = 0.0;
    accel.angular.y = 0.0;
    accel.angular.z = 0.0;
    next_point.accelerations.push_back(accel);

    desiredStatePub->publish(next_point);
    // ~~~~ end solution
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    // ~
    //                                 end part 1.3
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }

public:
  explicit WaypointFollower() : Node("waypoint_follower_node") {
    currentStateSub = this->create_subscription<nav_msgs::msg::Odometry>(
        "/current_state", 1,
        std::bind(&WaypointFollower::onCurrentState, this,
                  std::placeholders::_1));

    poseArraySub = this->create_subscription<geometry_msgs::msg::PoseArray>(
        "/desired_traj_vertices", 1,
        std::bind(&WaypointFollower::generateOptimizedTrajectory, this,
                  std::placeholders::_1));

    desiredStatePub = this->create_publisher<
        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>("/desired_state",
                                                            1);

    desiredStateTimer =
        rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(0.01),
                             std::bind(&WaypointFollower::publishDesiredState, this));
    desiredStateTimer->reset();
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaypointFollower>());
  return 0;
}
