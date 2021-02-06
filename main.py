"""
This program creates a single neuron neural network tuned to guess
if a point is above or below a randomly generated line.

The neuron is called Perceptron and has 3 inputs and weights to calculate its output
    input 1 is the X coordinate of the point
    Input 2 is the y coordinate of the point
    Input 3 is the bias and it is always 1

    Input 3 or the bias is required for lines that do not cross the origin (0,0)

The Perceptron starts with all weights equal to zero and learns
using 1,000 random points per each iteration.

The output of the perceptron is calculated based on the following
    if x * weight_x + y weight_y + weight_bias is positive then 1 else 0

The error for each point is calculated as the expected outcome of the perceptron minus the real outcome
therefore there are only 3 possible error values:

    Expected    Calculated  Error
    1           -1          1
    1           1           0
    -1          -1          0
    -1          1           -1

With every point that is learned if the error is not 0 the weights are adjusted according to:
    New_weight = Old_weight + error * input * learning_rate
    for example: New_weight_x = Old_weight_x + error * x * learning rate

The learning_rate decreases with every iteration as follows:
    learning_rate = 0.01 / (iteration + 1)
    this is important to ensure that once the weights are nearing the optimal values
    the adjustment in each iteration is subsequently more subtle

"""
import random as rnd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# number of points to learn per each animation frame or iteration
points_per_frame = 1000

# counter arrays to plot the errors and line parameters adjustments
error_A_data, error_B_data = [], []
sqr_error_data = []
weights_x_data, weights_y_data, weights_z_data = [], [], []

# initialize the counter for total learned points
total_learned_points = 0

# create the random original line
A = rnd.uniform(-10, 10)
B = rnd.uniform(-10, 10)
print("Randomly generated line:\n",
      "y = " + str(round(A, 5)) +
      "x + " + str(round(B, 5))
      )


# function to calculate f(x) based on the original line
def f(x):
    return A * x + B


# function to calculate f(x) based on the calculated line by the perceptron
def f1(x):
    return (-P.weights[0] / P.weights[1]) * x - (P.weights[2] / P.weights[1])


# define the Perceptron object
class Perceptron:
    def __init__(self):
        # initialize the 3 weights with zero
        self.weights = np.zeros(3)

    # calculate the estimated result based on inputs
    def guess(self, x_, y_):
        value = self.weights[0] * x_ + self.weights[1] * y_ + self.weights[2]
        if value >= 0:
            return 1
        else:
            return 0

    # adjust weights based on inputs and target value (label)
    def learn(self, inputs: [], label_, learning_rate=0.0001):
        error_ = label_ - self.guess(inputs[0], inputs[1])
        if error_:
            # only log the weights and A, B errors when the point presents error
            # ensure we do not divide by zero
            if self.weights[1]:
                weights_x_data.append(self.weights[0])
                weights_y_data.append(self.weights[1])
                weights_z_data.append(self.weights[2])
                error_A_data.append(A - (-self.weights[0] / self.weights[1]))
                error_B_data.append(B - (-self.weights[2] / self.weights[1]))
            # update the weights based on the error
            self.weights = self.weights + (learning_rate * error_) * np.array(inputs)
            # for i_ in range(len(self.weights)):
            #     self.weights[i_] += (inputs[i_] * error_ * learning_rate)

# create the Perceptron
P = Perceptron()

# initialize the four graphs to show during the learning iterations
fig, axs = plt.subplots(2, 2)
fig.canvas.set_window_title('Learning ' + str(points_per_frame) + ' points per iteration')
fig.set_size_inches(9, 6)
fig.suptitle('Approximating a line of form ' +
             str(round(A, 3)) +
             ' x + ' +
             str(round(B, 3)),
             fontsize=12
             )

# main function to be called at each animation frame
def animate(i_):
    # initialize the error counting variables per frame
    errores = 0
    error_total = 0

    # initialize an array to collect all the points to be plotted
    coords = []

    # clear the charts from the previous frame
    axs[0, 0].clear()
    axs[0, 1].clear()
    axs[1, 0].clear()
    axs[1, 1].clear()

    # initialize the header text variables for each chart
    line_error_text = axs[0, 0].text(0.01, 1.02, '', transform=axs[0, 0].transAxes)
    error_count_text = axs[0, 1].text(0.01, 1.02, '', transform=axs[0, 1].transAxes)
    error_A_text = axs[1, 0].text(0.01, 1.02, '', transform=axs[1, 0].transAxes)
    error_B_text = axs[1, 1].text(0.01, 1.02, '', transform=axs[1, 1].transAxes)

    # main loop to generate the points to be checked and learned
    for j in range(points_per_frame):
        # create a random point between x=-10, x=10 and y=-30, y=30
        x_ = rnd.uniform(-10, 10)
        y_ = rnd.uniform(-100, 100)
        # assign the target value or label for the point
        if y_ >= f(x_):
            label_ = 1
        else:
            label_ = 0
        # get the perceptron computation
        guess_ = P.guess(x_, y_)
        # calculate the error
        error_ = label_ - guess_
        # if there is an error, count it and add a red point to the plot
        if error_:
            coords.append([x_, y_, 'r'])
            errores += 1
            error_total += error_ ** 2
        # if there is not an error add a green point to the plot if the dot is above and blue if below
        else:
            if guess_:
                coords.append([x_, y_, 'g'])
            else:
                coords.append([x_, y_, 'b'])
        # call the function to correct the weights in the Perceptron based on the error
        # P.learn([x_, y_, 1], label_, 100 / (i_ + 1))
        if len(error_A_data) and len(error_B_data):
            learning_rate = ((error_A_data[-1]**2) + error_B_data[-1]**2)
        else:
            learning_rate = 1
        P.learn([x_, y_, 1],
                label_,
                learning_rate)
    # keep the count of total points being learned
    global total_learned_points
    total_learned_points += (i_ + 1) * points_per_frame
    # Set the text for the chart headers
    total_points_txt = "{points:,}"
    line_error_text.set_text(
        "Total learned points: " +
        total_points_txt.format(points=total_learned_points) +
        ", Learn_rate: " +
        str(round(learning_rate, 3))
    )
    error_count_text.set_text(
        'Model fit: ' +
        str(100 - errores/points_per_frame*100) +
        "%, #err: " +
        str(errores) +
        ", sum(err^2): " +
        str(error_total)
    )
    error_A_text.set_text(
        "Line approximation error A:" +
        str(round(A - (-P.weights[0] / P.weights[1]), 5))
    )
    error_B_text.set_text(
        "Line approximation error B:" +
        str(round(B - (-P.weights[2] / P.weights[1]), 5))
    )

    # plot dots (green for correct guess, red for incorrect guess)
    coords = np.transpose(coords)
    axs[0, 0].scatter(np.float64(coords[:][0]), np.float64(coords[:][1]), s=4, c=coords[:][2])

    # plot the original line with a width of 5 (should appear blue in the plot)
    axs[0, 0].plot([-10, 10], [f(-10), f(10)], lw=5)

    # plot the current learned line (should appear orange)
    axs[0, 0].plot([-10, 10], [f1(-10), f1(10)])

    # plot the squared error
    sqr_error_data.append(error_total)
    axs[0, 1].plot(sqr_error_data)

    # plot the approximation error for A
    axs[1, 0].plot(error_A_data)

    # plot the approximation error for B
    axs[1, 1].plot(error_B_data)


# start the animation; use the interval parameter to pause between iterations
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.ion = 0
plt.show()

# once the animation window is closed output to console the final line and error
total_points_txt = "{points:,}"
if P.weights[1]:
    print("\nFinal Perceptron approximated line after " +
          total_points_txt.format(points=total_learned_points) +
          " points learned:\n",
          "y = " + str(round(-P.weights[0] / P.weights[1], 5)) +
          "x + " + str(round(-P.weights[2] / P.weights[1], 5))
          )
    print("\nFinal line error:\n" +
          "A: " +
          str(round(A - (-P.weights[0] / P.weights[1]), 5)) +
          " B: " +
          str(round(B - (-P.weights[2] / P.weights[1]), 5))
          )

# create a new 3d chart with the movement of the 3 weights
fig2 = plt.figure()
fig2.suptitle('Adjustment of weights through time', fontsize=12)
fig2.canvas.set_window_title('Learned using '
                             + total_points_txt.format(points=total_learned_points)
                             + ' points')
fig2.set_size_inches(9, 6)
ax = fig2.gca(projection='3d')
ax.plot(weights_x_data, weights_y_data, weights_z_data)
ax.set_xlabel('Wx')
ax.set_ylabel('Wy')
ax.set_zlabel('Wb')
plt.show()
