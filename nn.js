function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
  //sigmoid(x)*(1-sigmoid(x))
  return x * (1 - x);
}

class NeuronalNetwork {

  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weighths_ih = new Matrix(this.input_nodes, this.hidden_nodes);
    this.weighths_ho = new Matrix(this.hidden_nodes, this.output_nodes);
    this.weighths_ih.randomize();
    this.weighths_ho.randomize();

    this.bias_h = new Matrix(1, this.hidden_nodes);
    this.bias_o = new Matrix(1, this.output_nodes);

    this.bias_h.randomize();
    this.bias_o.randomize();
    this.lr = 0.1;
  }

  predict(input_array) {
    // Calculate Hidden layer
    let input = Matrix.fromArray(input_array);
    let hidden = Matrix.dot(input, this.weighths_ih);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);

    // Calculate Output layer
    let output = Matrix.dot(hidden, this.weighths_ho);
    output.add(this.bias_o);
    output.map(sigmoid);

    return output.toArray();
  }

  train(input_array, expected_output) {
    //Feedfoward
    let input = Matrix.fromArray(input_array);
    let hidden = Matrix.dot(input, this.weighths_ih);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);

    // Calculate Output layer
    let output = Matrix.dot(hidden, this.weighths_ho);
    output.add(this.bias_o);
    output.map(sigmoid);


    // Convert array to Matrix objects
    let y = Matrix.fromArray(expected_output);
    // Calculate error
    let output_errors = Matrix.subtract(y, output);
    //Gradient

    //OUTPUT
    let output_gradient = Matrix.map(output, dsigmoid);

    output_gradient.multiply(output_errors);
    output_gradient.multiply(this.lr);
    let hidden_T = Matrix.transpose(hidden);

    let weighths_ho_deltas = Matrix.dot(hidden_T, output_gradient);
    this.weighths_ho.add(weighths_ho_deltas);
    this.bias_o.add(output_gradient);



    //
    // // //HIDDEN
    let who_t = Matrix.transpose(this.weighths_ho);
    let hidden_errors = Matrix.dot(output_errors, who_t);
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.lr);

    let input_T = Matrix.transpose(input);
    let weighths_ih_deltas = Matrix.dot(input_T, hidden_gradient);
    this.weighths_ih.add(weighths_ih_deltas);
    this.bias_h.add(hidden_gradient);

  }




}