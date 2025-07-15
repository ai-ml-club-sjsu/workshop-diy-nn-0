impl Layer for Linear{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=cache.len()-outputgrad.len()..;
		cache.drain(range).zip(outputgrad).map(|(i,og)|{
			todo!()						//TODO accumulate the parameter gradients given input i and output gradient og, then end the block with the input gradient
		}).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		if let Some(c)=cache{c.extend(input.iter().cloned())}
		let output=input.into_iter().map(|i|{
			todo!()						//TODO matrix multiply the weight by the input, and then add bias if you have it
		}).collect();
		output
	}
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value)){
		todo!()							//TODO apply the optimization function f to each parameter and it's gradient, then set the parameter and gradient to the returned value afterwards
	}
}
impl Layer for Tanh{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=cache.len()-outputgrad.len()..;
		cache.drain(range).zip(outputgrad).map(|(o,g)|{
			todo!()						//TODO this layer has no parameter so you only need to find the input gradient. given y=tanh(x), dy/dx = 1.0-y^2. conveniently, since the outputs were cached rather than the inputs, you have an output o and output gradient g, so the input gradient for each component at position n will be (1.0-o_n*o_n)*g_n The map_2 function I've implemented for Value is convenient for applying a componentwise function with two arguments.
		}).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		let output:Vec<Value>=input.into_iter().map(|i|{
			todo!()						//TODO apply tanh to each component. The map function I've implemented for Value is convenient for applying a componentwise function
		}).collect();
		if let Some(c)=cache{c.extend(output.iter().cloned())}
		output
	}
	fn opt_step(&mut self,_f:&mut dyn FnMut(Value,Value)->(Value,Value)){
										// nothing to do here since the tanh layer has no parameter
	}
}
impl Linear{
	pub fn new(inputdim:usize,outputdim:usize)->Self{
		todo!()							//TODO randomly initialize the weights using Value::random. Initialize bias too if you decide to put it
	}
}
impl Tanh{
	pub fn new()->Self{Tanh}
}
#[derive(Clone,Debug)]
/// linear (matmul) layer
pub struct Linear{
	weight:Value						//TODO what else do we need in here
}
#[derive(Clone,Debug)]
/// tanh layer
pub struct Tanh;
/// basic nn layer trait
pub trait Layer{
	/// applies the backward pass operation
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>;
	/// applies the forward pass operation
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>;
	/// applies the inference pass operation
	fn infer(&mut self,input:Vec<Value>)->Vec<Value>{self.forward(&mut None,input)}
	/// applys sgd optimization
	fn momentum_sgd(&mut self,lr:f32,persistence:f32){self.opt_step(&mut |param,paramgrad|(param.map_2(|p,g|g*lr+p,paramgrad.clone()),paramgrad*persistence))}
	/// adjusts the parameters according to the optimization function (param, gradient) -> new param, new gradient. this should call the function a consitent number of times between backward steps
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value));
}
use crate::value::Value;
use std::mem::take;
