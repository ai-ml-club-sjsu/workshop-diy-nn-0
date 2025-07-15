impl Layer for Linear{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=cache.len()-outputgrad.len()..;
		cache.drain(range).zip(outputgrad).map(|(i,og)|{
			let db=og.clone();
			let di=og.clone().matmul(self.weight.clone().transpose());
			let dw=i.transpose().matmul(og);

			if let Some(weightgrad)=&mut self.weightgrad{*weightgrad+=dw}else{self.weightgrad=Some(dw)}
			if self.bias.is_some(){
				if let Some(biasgrad)=&mut self.biasgrad{*biasgrad+=db}else{self.biasgrad=Some(db)}
			}
			di
		}).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		if let Some(c)=cache{c.extend(input.iter().cloned())}
		let output=input.into_iter().map(|i|{
			let mut o=i.matmul(self.weight.clone());
			if let Some(bias)=&self.bias{o+=bias}
			o
		}).collect();
		output
	}
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value)){
		if let (weight,Some(weightgrad))=(&mut self.weight,&mut self.weightgrad){
			for _ in 0..weightgrad.dims().len().saturating_sub(weight.dims().len()){*weightgrad=take(weightgrad).avg_dim(0).squeeze(0,1)}
			let (w,g)=f(take(weight),take(weightgrad));
			(self.weight,self.weightgrad)=(w,Some(g));
		}else{
			f(Value::default(),Value::default());
		}
		if let (Some(bias),Some(biasgrad))=(&mut self.bias,&mut self.biasgrad){
			for _ in 0..biasgrad.dims().len().saturating_sub(bias.dims().len()){*biasgrad=take(biasgrad).avg_dim(0).squeeze(0,1)}
			let (b,g)=f(take(bias),take(biasgrad));
			(self.bias,self.biasgrad)=(Some(b),Some(g));
		}else{
			f(Value::default(),Value::default());
		}
	}
}
impl Layer for Tanh{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=cache.len()-outputgrad.len()..;
		cache.drain(range).zip(outputgrad).map(|(o,g)|o.map_2(|o,g|(1.0-o*o)*g,g)).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		let output:Vec<Value>=input.into_iter().map(|i|i.map(f32::tanh)).collect();
		if let Some(c)=cache{c.extend(output.iter().cloned())}
		output
	}
	fn opt_step(&mut self,_f:&mut dyn FnMut(Value,Value)->(Value,Value)){}
}
impl Linear{
	pub fn new(bias:bool,inputdim:usize,outputdim:usize)->Self{
		let bias=bias.then(||Value::random([outputdim]));
		let biasgrad=None;
		let weight=Value::random([inputdim,outputdim]);
		let weightgrad=None;
		Self{bias,biasgrad,weight,weightgrad}
	}
}
impl Tanh{
	pub fn new()->Self{Tanh}
}
#[derive(Clone,Debug)]
/// linear (matmul) layer
pub struct Linear{bias:Option<Value>,biasgrad:Option<Value>,weight:Value,weightgrad:Option<Value>}
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
