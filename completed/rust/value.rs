/// gets data from the arc buffer
fn buffer<'a,E>(buffer:&'a Arc<[E]>,count:usize,offset:&usize)->&'a [E]{&buffer[*offset..count+*offset]}
/// gets mutable data from the arc buffer
fn buffer_io<'a,E:Clone+Default>(buffer:&'a mut Arc<[E]>,count:usize,offset:&mut usize,outputcount:usize,outputoffset:&mut usize)->(&'a [E],&'a mut [E]){
	if Arc::strong_count(&*buffer)==1{
		if *offset>=outputcount{
			let b=Arc::get_mut(buffer).unwrap();
			let (o,i)=b.split_at_mut(*offset);
			*outputoffset=0;
			return (&i[..count],&mut o[..outputcount]);
		}
		if *offset+count+outputcount<=buffer.len(){
			let b=Arc::get_mut(buffer).unwrap();
			let (i,o)=b.split_at_mut(*offset+count);
			*outputoffset=*offset+count;
			return (&i[*offset..],&mut o[..outputcount]);
		}
	}
	*buffer=buffer[*offset..count+*offset].iter().cloned().chain((0..outputcount).map(|_|E::default())).collect();
	*offset=0;
	*outputoffset=count;
	let (i,o)=Arc::get_mut(buffer).unwrap().split_at_mut(count);
	(&*i,o)
}
/// gets mutable data from the arc buffer
fn buffer_mut<'a,E:Clone>(buffer:&'a mut Arc<[E]>,count:usize,offset:&mut usize)->&'a mut [E]{
	if Arc::strong_count(&*buffer)==1{return &mut Arc::get_mut(buffer).unwrap()[*offset..count+*offset]}
	let b=Arc::from(&buffer[*offset..count+*offset]);
	*buffer=b;
	*offset=0;
	Arc::get_mut(buffer).unwrap()
}
/// computes the index of the component given the coordinates and strides
fn compute_index<P:AsRef<[usize]>,S:AsRef<[usize]>>(position:P,strides:S)->usize{position.as_ref().iter().rev().zip(strides.as_ref().iter().rev()).map(|(x,s)|x*s).sum()}
impl Add<&Value> for &Value{
	fn add(self,rhs:&Value)->Self::Output{self.clone().map_2(Add::add,rhs.clone())}
	type Output=Value;
}
impl Add<&Value> for &f32{
	fn add(self,rhs:&Value)->Self::Output{rhs.clone().map(|x|self+x)}
	type Output=Value;
}
impl Add<&Value> for f32{
	fn add(self,rhs:&Value)->Self::Output{rhs.clone().map(|x|self+x)}
	type Output=Value;
}
impl Add<&Value> for Value{
	fn add(self,rhs:&Value)->Self::Output{self.map_2(Add::add,rhs.clone())}
	type Output=Value;
}
impl Add<&f32> for &Value{
	fn add(self,rhs:&f32)->Self::Output{self.clone().map(|x|x+rhs)}
	type Output=Value;
}
impl Add<&f32> for Value{
	fn add(self,rhs:&f32)->Self::Output{self.map(|x|x+rhs)}
	type Output=Value;
}
impl Add<Value> for &Value{
	fn add(self,rhs:Value)->Self::Output{self.clone().map_2(Add::add,rhs)}
	type Output=Value;
}
impl Add<Value> for Value{
	fn add(self,rhs:Value)->Self::Output{self.map_2(Add::add,rhs)}
	type Output=Value;
}
impl Add<Value> for &f32{
	fn add(self,rhs:Value)->Self::Output{rhs.map(|x|self+x)}
	type Output=Value;
}
impl Add<Value> for f32{
	fn add(self,rhs:Value)->Self::Output{rhs.map(|x|self+x)}
	type Output=Value;
}
impl Add<f32> for &Value{
	fn add(self,rhs:f32)->Self::Output{self.clone().map(|x|x+rhs)}
	type Output=Value;
}
impl Add<f32> for Value{
	fn add(self,rhs:f32)->Self::Output{self.map(|x|x+rhs)}
	type Output=Value;
}
impl AddAssign<&Value> for Value{
	fn add_assign(&mut self,rhs:&Value){*self=take(self)+rhs}
}
impl AddAssign<Value> for Value{
	fn add_assign(&mut self,rhs:Value){*self=take(self)+rhs}
}
impl AsRef<[usize]> for Shape{
	fn as_ref(&self)->&[usize]{&self.dims}
}
impl AsRef<Self> for Shape{
	fn as_ref(&self)->&Self{self}
}
impl Coordinates{
	fn from_mem(dims:Arc<[usize]>,mut mem:Arc<[usize]>)->Self{
		assert_eq!(dims.len(),mem.len());
		buffer_mut(&mut mem,dims.len(),&mut 0).fill(0);
		Self{dims,positions:mem,state:0}
	}
}
impl Deref for Shape{
	fn deref(&self)->&Self::Target{&self.dims}
	type Target=[usize];
}
impl Iterator for Coordinates{
	fn next(&mut self)->Option<Self::Item>{
		let dims=&self.dims;
		let positions=buffer_mut(&mut self.positions,dims.len(),&mut 0);
		let state=&mut self.state;

		if *state==0{
			*state=1;
			return Some(self.positions.clone());
		}else if *state==2{
			return None;
		}

		if dims.iter().rev().zip(positions.iter_mut().rev()).all(|(d,x)|{
			*x+=1;
			let nextline=d<=x;
			if nextline{*x=0}
			nextline
		}){
			*state=2;
			None
		}else{
			Some(self.positions.clone())
		}
	}
	type Item=Arc<[usize]>;
}
impl Mul<&Value> for &Value{
	fn mul(self,rhs:&Value)->Self::Output{self.clone().map_2(Mul::mul,rhs.clone())}
	type Output=Value;
}
impl Mul<&Value> for &f32{
	fn mul(self,rhs:&Value)->Self::Output{rhs.clone().map(|x|self*x)}
	type Output=Value;
}
impl Mul<&Value> for f32{
	fn mul(self,rhs:&Value)->Self::Output{rhs.clone().map(|x|self*x)}
	type Output=Value;
}
impl Mul<&Value> for Value{
	fn mul(self,rhs:&Value)->Self::Output{self.map_2(Mul::mul,rhs.clone())}
	type Output=Value;
}
impl Mul<&f32> for &Value{
	fn mul(self,rhs:&f32)->Self::Output{self.clone().map(|x|x*rhs)}
	type Output=Value;
}
impl Mul<Value> for &Value{
	fn mul(self,rhs:Value)->Self::Output{self.clone().map_2(Mul::mul,rhs)}
	type Output=Value;
}
impl Mul<Value> for Value{
	fn mul(self,rhs:Value)->Self::Output{self.map_2(Mul::mul,rhs)}
	type Output=Value;
}
impl Mul<Value> for &f32{
	fn mul(self,rhs:Value)->Self::Output{rhs.map(|x|self*x)}
	type Output=Value;
}
impl Mul<Value> for f32{
	fn mul(self,rhs:Value)->Self::Output{rhs.map(|x|self*x)}
	type Output=Value;
}
impl Mul<f32> for &Value{
	fn mul(self,rhs:f32)->Self::Output{self.clone().map(|x|x*rhs)}
	type Output=Value;
}
impl Mul<f32> for Value{
	fn mul(self,rhs:f32)->Self::Output{self.map(|x|x*rhs)}
	type Output=Value;
}
impl Neg for &Value{
	fn neg(self)->Self::Output{self.clone().map(Neg::neg)}
	type Output=Value;
}
impl Neg for Value{
	fn neg(self)->Self::Output{self.map(Neg::neg)}
	type Output=Value;
}
impl Shape{
	/// applies broadcasting
	pub fn broadcast(mut self,dims:&[usize])->Shape{
		let l=dims.len();
		let n=l.saturating_sub(self.len());
		self=self.unsqueeze(0,n);
		let skip=self.len().saturating_sub(l);
		let (d,s)=(buffer_mut(&mut self.dims,l,&mut 0),buffer_mut(&mut self.strides,l,&mut 0));

		d[skip..].iter().zip(dims.iter()).filter(|(d,b)|d!=b).for_each(|(&d,&b)|if d!=1&&b!=1{panic!("{d:?} cannot be broadcast to {dims:?}")});
		d[skip..].iter_mut().zip(dims.iter()).zip(s.iter_mut()).filter(|((d,b),_s)|d!=b).for_each(|((d,b),s)|if *d==1{(*d,*s)=(*b,0)});

		self
	}
	/// removes the 1 dimensions in range d..d+n
	pub fn squeeze(self,d:usize,n:usize)->Shape{
		assert!(d+n<=self.len());
		if n==0{return self}
		let (dims,strides)=(&self.dims,&self.strides);
		let strides=strides[..d].iter().copied().chain(strides[d+n..].iter().copied().zip(dims[d..d+1].iter().copied()).filter(|&(_s,d)|d!=1).map(|(s,_d)|s)).chain(strides[d+n..].iter().copied()).collect();
		let dims=dims[..d].iter().copied().chain(dims[d..d+n].iter().copied().filter(|&d|d!=1)).chain(dims[d+n..].iter().copied()).collect();
		Shape{dims,strides}
	}
	/// adds n 1 dimensions at postion d
	pub fn unsqueeze(self,d:usize,n:usize)->Shape{
		assert!(d<=self.len());
		if n==0{return self}
		let (dims,strides)=(&self.dims,&self.strides);
		let dims=dims[..d].iter().copied().chain((0..n).map(|_|1)).chain(dims[d..].iter().copied()).collect();
		let strides=strides[..d].iter().copied().chain((0..n).map(|_|0)).chain(strides[d..].iter().copied()).collect();
		Shape{dims,strides}
	}
	/// swaps two of the dimensions
	pub fn swap_dims(mut self,a:usize,b:usize)->Shape{
		if a==b{return self}
		let l=self.dims.len();
		let (d,s)=(buffer_mut(&mut self.dims,l,&mut 0),buffer_mut(&mut self.strides,l,&mut 0));
		d.swap(a,b);
		s.swap(a,b);
		self
	}
	/// swaps the last two dimensions
	pub fn transpose(self)->Shape{
		let l=self.dims.len();
		if l==0{return self}
		if l==1{return self.unsqueeze(1,1)}
		self.swap_dims(l-1,l-2)
	}
}
impl Sub<&Value> for &Value{
	fn sub(self,rhs:&Value)->Self::Output{self.clone().map_2(Sub::sub,rhs.clone())}
	type Output=Value;
}
impl Sub<&Value> for Value{
	fn sub(self,rhs:&Value)->Self::Output{self.map_2(Sub::sub,rhs.clone())}
	type Output=Value;
}
impl Sub<Value> for &Value{
	fn sub(self,rhs:Value)->Self::Output{self.clone().map_2(Sub::sub,rhs)}
	type Output=Value;
}
impl Sub<Value> for Value{
	fn sub(self,rhs:Value)->Self::Output{self.map_2(Sub::sub,rhs)}
	type Output=Value;
}
impl Value{
	/// averages along the dimension
	pub fn avg_dim(mut self,d:usize)->Value{
		let l=self.dims().len();
		assert!(d<l);
		self=self.swap_dims(0,d);
		let (dim,stride)=(self.shape.dims[0],self.shape.strides[0]);
		let data=buffer_mut(&mut self.buffer,self.count,&mut self.offset);
		Coordinates::from_mem(self.shape.dims.clone(),self.shape.dims.clone()).take(self.shape.dims[1..].iter().product()).for_each(|c|{
			let offset=compute_index(c,&self.shape.strides);
			let mut acc=0.0;
			for n in 0..dim{acc+=data[n*stride+offset]}
			data[offset]=acc/dim as f32;
		});
		buffer_mut(&mut self.shape.dims,l,&mut 0)[0]=1;
		self.count=self.shape.dims.iter().zip(self.shape.strides.iter()).map(|(d,s)|d*s).max().unwrap_or(1);
		self.swap_dims(0,d)
	}
	/// applies broadcasting
	pub fn broadcast(mut self,dims:&[usize])->Value{
		self.shape=self.shape.broadcast(dims);
		self
	}
	/// references the dimensions
	pub fn dims(&self)->&[usize]{&self.shape.dims}
	/// creates from data and dimensions
	pub fn from_data<A:AsRef<[f32]>,B:AsRef<[usize]>>(data:A,dims:B)->Self{
		let (data,dims)=(data.as_ref(),dims.as_ref());
		let mut strides=vec![0;dims.len()];
		let offset=0;

		let count=dims.iter().zip(strides.iter_mut()).rev().fold(1,|product,(&dim,stride)|{
			*stride=product;
			product*dim
		});
		let (buffer,dims,strides)=(Arc::from(data),Arc::from(dims),Arc::from(strides));
		let shape=Shape{dims,strides};
		Self{buffer,count,offset,shape}
	}
	/// converts into the inner data
	pub fn into_data(self)->Vec<f32>{
		let (dims,strides)=(self.shape.dims,self.shape.strides);
		let data=buffer(&self.buffer,self.count,&self.offset);
		let positions=Arc::from(vec![0;dims.len()]);
		Coordinates::from_mem(dims,positions).map(|c|data[compute_index(c,&strides)]).collect()
	}
	/// applies the function to each component
	pub fn map<F:Fn(f32)->f32>(mut self,f:F)->Value{
		buffer_mut(&mut self.buffer,self.count,&mut self.offset).iter_mut().for_each(|x|*x=f(*x));
		self
	}
	/// applies the function to each pair of components
	pub fn map_2<F:Fn(f32,f32)->f32>(self,f:F,r:Value)->Value{
		let mut l=self.broadcast(r.dims());
		let mut r=r.broadcast(l.dims());
		let (lbcnt,rbcnt)=((Arc::strong_count(&l.buffer)>1 as usize,l.count),(Arc::strong_count(&r.buffer)>1 as usize,r.count));

		if l.shape.strides==r.shape.strides{
			if lbcnt<rbcnt{
				buffer_mut(&mut l.buffer,l.count,&mut l.offset).iter_mut().zip(r.buffer[r.offset..r.count+r.offset].iter()).for_each(|(l,r)|*l=f(*l,*r));
				Value{buffer:l.buffer,count:l.count,offset:l.offset,shape:l.shape}
			}else{
				l.buffer[l.offset..l.count+l.offset].iter().zip(buffer_mut(&mut r.buffer,r.count,&mut r.offset).iter_mut()).for_each(|(l,r)|*r=f(*l,*r));
				Value{buffer:r.buffer,count:r.count,offset:r.offset,shape:r.shape}
			}
		}else{// TODO this doesn't necessarily work and should probably use buffer io pattern
			let lb=buffer_mut(&mut l.buffer,l.count,&mut l.offset);
			let rb=&r.buffer[r.offset..r.offset+r.count];
			Coordinates::from_mem(l.shape.dims.clone(),r.shape.dims).for_each(|c|{
				let (lx,rx)=(compute_index(&c,&l.shape.strides),compute_index(&c,&r.shape.strides));
				lb[lx]=f(lb[lx],rb[rx]);
			});
			Value{buffer:l.buffer,count:l.count,offset:l.offset,shape:l.shape}
		}
	}
	#[track_caller]
	/// applies matrix multiplication
	pub fn matmul(self,r:Value)->Value{// TODO finish and test
		let (ld,rd)=(self.dims().len(),r.dims().len());
		let d=ld.max(rd).max(2);
		let mut l=self.unsqueeze(0,d.saturating_sub(ld));
		let mut r=r.unsqueeze(0,d.saturating_sub(rd));
		let (ldims,rdims)=(buffer_mut(&mut l.shape.dims,d,&mut 0),buffer_mut(&mut r.shape.dims,d,&mut 0));
		let (lstrides,rstrides)=(buffer_mut(&mut l.shape.strides,d,&mut 0),buffer_mut(&mut r.shape.strides,d,&mut 0));
		let (ycols,yrows)=(rdims[d-1],ldims[d-2]);
		let shared=ldims[d-1];

		assert!(rdims[d-2]==shared,"a {yrows}x{shared} matrix cannot be multiplied with a {}x{ycols} matrix",rdims[d-2]);
		ldims[..d-2].iter().zip(rdims[..d-2].iter()).filter(|(d,b)|d!=b).for_each(|(&d,&b)|if d!=1&&b!=1{panic!("{:?} cannot be broadcast to {:?}",&ldims[..d-2],&rdims[..d-2])});
		ldims[..d-2].iter_mut().zip(lstrides[..d-2].iter_mut()).zip(rdims[..d-2].iter_mut()).zip(rstrides[..d-2].iter_mut()).for_each(|(((ld,ls),rd),rs)|{
			if *ld==1{*ls=0}
			if *rd==1{(*rd,*rs)=(*ld,0)}
			*ld=0;
		});
		(rdims[d-1],rdims[d-2])=(ycols,yrows);

		let (dims,lstrides,rstrides)=(&rdims[..d-2],&l.shape.strides[..d-2],&r.shape.strides[..d-2]);
		let (lcolstride,lrowstride,rcolstride,rrowstride)=(l.shape.strides[d-2],l.shape.strides[d-1],r.shape.strides[d-2],r.shape.strides[d-1]);
		let mut yoff=0;
		let position=&mut ldims[..d-2];
		let rdata=buffer(&r.buffer,r.count,&r.offset);
		let ycount:usize=rdims.iter().product();
		let (ldata,ydata)=buffer_io(&mut l.buffer,l.count,&mut l.offset.clone(),ycount,&mut l.offset);

		loop{
			let (ldata,rdata)=(&ldata[compute_index(&*position,lstrides)..],&rdata[compute_index(&*position,rstrides)..]);
			for col in 0..ycols{
				let rdata=&rdata[col*rrowstride..];
				for row in 0..yrows{
					let ldata=&ldata[lcolstride*row..];
					let mut acc=0.0;
					for n in 0..shared{acc+=ldata[lrowstride*n]*rdata[rcolstride*n]}
					ydata[yoff]=acc;
					yoff+=1;
				}
			}

			if dims.iter().rev().zip(position.iter_mut().rev()).all(|(d,x)|{
				let nextline=d==x;
				if nextline{*x=0}else{*x+=1}
				nextline
			}){
				break
			}
		}

		let mut y=l;
		y.count=ycount;
		y.shape.dims=r.shape.dims;
		y.shape.dims.iter().zip(buffer_mut(&mut y.shape.strides,d,&mut 0)).rev().fold(1,|product,(&dim,stride)|{
			*stride=product;
			product*dim
		});
		if ld<1&&d==2{y.squeeze(0,2)}else{y}
	}
	/// creates a random tensor
	pub fn random<A:AsRef<[usize]>>(dims:A)->Self{
		let mut seed=tseed();
		let data:Vec<f32>=(0..dims.as_ref().iter().product()).map(|_|rfloat(&mut seed)).collect();

		Self::from_data(data,dims)
	}
	/// returns the tensor shape, which contains the dimensions
	pub fn shape(&self)->Shape{self.shape.clone()}
	/// removes the 1 dimensions in range d..d+n
	pub fn squeeze(mut self,d:usize,n:usize)->Value{
		self.shape=self.shape.squeeze(d,n);
		self
	}
	/// swaps two of the dimensions
	pub fn swap_dims(mut self,a:usize,b:usize)->Value{
		self.shape=self.shape.swap_dims(a,b);
		self
	}
	/// swaps the last two dimensions
	pub fn transpose(mut self)->Value{
		self.shape=self.shape.transpose();
		self
	}
	/// adds n 1 dimensions at postion d
	pub fn unsqueeze(mut self,d:usize,n:usize)->Value{
		self.shape=self.shape.unsqueeze(d,n);
		self
	}
}
/// prng float on >=-1 and <1
pub fn rfloat(seed:&mut u128)->f32{
	update_seed(seed);
	*seed as i32 as f32*2.0_f32.powi(-31)
}
pub fn tseed()->u128{
	let mut seed=SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
	update_seed(&mut seed);
	update_seed(&mut seed);
	update_seed(&mut seed);
	seed
}/// prng seed updating
pub fn update_seed(seed:&mut u128){
	*seed^=*seed<<15;
	*seed^=*seed>>4;
	*seed^=*seed<<21;
}
#[derive(Clone,Debug,Default)]
/// iterator over tensor positions
pub struct Coordinates{dims:Arc<[usize]>,positions:Arc<[usize]>,state:usize}
#[derive(Clone,Debug,Default)]
/// tensor value dimensions
pub struct Shape{dims:Arc<[usize]>,strides:Arc<[usize]>}
#[derive(Clone,Debug,Default)]
/// nn layer io value. stores data and dimensions
pub struct Value{buffer:Arc<[f32]>,count:usize,offset:usize,shape:Shape}
use std::{
	mem::take,ops::{Add,AddAssign,Deref,Mul,Neg,Sub},sync::Arc,time::SystemTime
};
