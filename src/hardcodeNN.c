#include <stdio.h>
#include <comp_graph.h>
#include <stdlib.h>

int main() {
node input0_;
node* prevNodesinput0_ =NULL;
input0_.name = "input0_";
input0_.ref_num = 2;
input0_.value = 0.000000;
input0_.prev = &prevNodesinput0_;
input0_.c = 0;
input0_.curr_c = 0;
double actParamsinput0_[2] ={1.000000,0};
input0_.params = actParamsinput0_;
input0_.n = 2;
input0_.visited_front = 0;
input0_.act_func = linear;


node weight0_0_;
node* prevNodesweight0_0_[1] ={&input0_};
weight0_0_.name = "weight0_0_"; 
weight0_0_.ref_num = 1;
weight0_0_.value = 0.000000;
weight0_0_.prev = &prevNodesweight0_0_;
weight0_0_.c = 1;
weight0_0_.curr_c = 1;
double actParamsweight0_0_[2] ={1.000000,0};
weight0_0_.params = actParamsweight0_0_;
weight0_0_.n = 2;
weight0_0_.visited_front = 1;
weight0_0_.act_func = linear;


node input1_;
node* prevNodesinput1_ =NULL;
input1_.name = "input1_";
input1_.ref_num = 2;
input1_.value = 0.000000;
input1_.prev = &prevNodesinput1_;
input1_.c = 0;
input1_.curr_c = 0;
double actParamsinput1_[2] ={1.000000,0};
input1_.params = actParamsinput1_;
input1_.n = 2;
input1_.visited_front = 0;
input1_.act_func = linear;


node weight0_1_;
node* prevNodesweight0_1_[1] ={&input1_};
weight0_1_.name = "weight0_1_"; 
weight0_1_.ref_num = 1;
weight0_1_.value = 0.000000;
weight0_1_.prev = &prevNodesweight0_1_;
weight0_1_.c = 1;
weight0_1_.curr_c = 1;
double actParamsweight0_1_[2] ={2.000000,0};
weight0_1_.params = actParamsweight0_1_;
weight0_1_.n = 2;
weight0_1_.visited_front = 1;
weight0_1_.act_func = linear;


node input2_;
node* prevNodesinput2_ =NULL;
input2_.name = "input2_";
input2_.ref_num = 2;
input2_.value = 0.000000;
input2_.prev = &prevNodesinput2_;
input2_.c = 0;
input2_.curr_c = 0;
double actParamsinput2_[2] ={1.000000,0};
input2_.params = actParamsinput2_;
input2_.n = 2;
input2_.visited_front = 0;
input2_.act_func = linear;


node weight0_2_;
node* prevNodesweight0_2_[1] ={&input2_};
weight0_2_.name = "weight0_2_"; 
weight0_2_.ref_num = 1;
weight0_2_.value = 0.000000;
weight0_2_.prev = &prevNodesweight0_2_;
weight0_2_.c = 1;
weight0_2_.curr_c = 1;
double actParamsweight0_2_[2] ={3.000000,0};
weight0_2_.params = actParamsweight0_2_;
weight0_2_.n = 2;
weight0_2_.visited_front = 1;
weight0_2_.act_func = linear;


node bias0_;
node* prevNodesbias0_[3] ={&weight0_0_,&weight0_1_,&weight0_2_};
bias0_.name = "bias0_"; 
bias0_.ref_num = 1;
bias0_.value = 0.000000;
bias0_.prev = &prevNodesbias0_;
bias0_.c = 3;
bias0_.curr_c = 3;
double actParamsbias0_[2] ={1.000000,0};
bias0_.params = actParamsbias0_;
bias0_.n = 2;
bias0_.visited_front = 1;
bias0_.act_func = linear;


node actfunc0_;
node* prevNodesactfunc0_[1] ={&bias0_};
actfunc0_.name = "actfunc0_"; 
actfunc0_.ref_num = 2;
actfunc0_.value = 0.000000;
actfunc0_.prev = &prevNodesactfunc0_;
actfunc0_.c = 1;
actfunc0_.curr_c = 1;
double actParamsactfunc0_[2] ={0.010000,0};
actfunc0_.params = actParamsactfunc0_;
actfunc0_.n = 2;
actfunc0_.visited_front = 1;
actfunc0_.act_func = leakyReLU;


node weight1_0_;
node* prevNodesweight1_0_[1] ={&input0_};
weight1_0_.name = "weight1_0_"; 
weight1_0_.ref_num = 1;
weight1_0_.value = 0.000000;
weight1_0_.prev = &prevNodesweight1_0_;
weight1_0_.c = 1;
weight1_0_.curr_c = 1;
double actParamsweight1_0_[2] ={1.000000,0};
weight1_0_.params = actParamsweight1_0_;
weight1_0_.n = 2;
weight1_0_.visited_front = 1;
weight1_0_.act_func = linear;


node weight1_1_;
node* prevNodesweight1_1_[1] ={&input1_};
weight1_1_.name = "weight1_1_"; 
weight1_1_.ref_num = 1;
weight1_1_.value = 0.000000;
weight1_1_.prev = &prevNodesweight1_1_;
weight1_1_.c = 1;
weight1_1_.curr_c = 1;
double actParamsweight1_1_[2] ={2.000000,0};
weight1_1_.params = actParamsweight1_1_;
weight1_1_.n = 2;
weight1_1_.visited_front = 1;
weight1_1_.act_func = linear;


node weight1_2_;
node* prevNodesweight1_2_[1] ={&input2_};
weight1_2_.name = "weight1_2_"; 
weight1_2_.ref_num = 1;
weight1_2_.value = 0.000000;
weight1_2_.prev = &prevNodesweight1_2_;
weight1_2_.c = 1;
weight1_2_.curr_c = 1;
double actParamsweight1_2_[2] ={3.000000,0};
weight1_2_.params = actParamsweight1_2_;
weight1_2_.n = 2;
weight1_2_.visited_front = 1;
weight1_2_.act_func = linear;


node bias1_;
node* prevNodesbias1_[3] ={&weight1_0_,&weight1_1_,&weight1_2_};
bias1_.name = "bias1_"; 
bias1_.ref_num = 1;
bias1_.value = 0.000000;
bias1_.prev = &prevNodesbias1_;
bias1_.c = 3;
bias1_.curr_c = 3;
double actParamsbias1_[2] ={1.000000,0};
bias1_.params = actParamsbias1_;
bias1_.n = 2;
bias1_.visited_front = 1;
bias1_.act_func = linear;


node actfunc1_;
node* prevNodesactfunc1_[1] ={&bias1_};
actfunc1_.name = "actfunc1_"; 
actfunc1_.ref_num = 2;
actfunc1_.value = 0.000000;
actfunc1_.prev = &prevNodesactfunc1_;
actfunc1_.c = 1;
actfunc1_.curr_c = 1;
double actParamsactfunc1_[2] ={0.010000,0};
actfunc1_.params = actParamsactfunc1_;
actfunc1_.n = 2;
actfunc1_.visited_front = 1;
actfunc1_.act_func = leakyReLU;


node output0_;
node* prevNodesoutput0_[2] ={&actfunc0_,&actfunc1_};
output0_.name = "output0_"; 
output0_.ref_num = 1;
output0_.value = 0.000000;
output0_.prev = &prevNodesoutput0_;
output0_.c = 2;
output0_.curr_c = 2;
double actParamsoutput0_[2] ={1.000000,0};
output0_.params = actParamsoutput0_;
output0_.n = 2;
output0_.visited_front = 1;
output0_.act_func = linear;


node loss0_;
node* prevNodesloss0_[1] ={&output0_};
loss0_.name = "loss0_"; 
loss0_.ref_num = 0;
loss0_.value = 0.000000;
loss0_.prev = &prevNodesloss0_;
loss0_.c = 1;
loss0_.curr_c = 1;
double actParamsloss0_[1] ={0};
loss0_.params = actParamsloss0_;
loss0_.n = 1;
loss0_.visited_front = 1;
loss0_.act_func = mse;


node output1_;
node* prevNodesoutput1_[2] ={&actfunc0_,&actfunc1_};
output1_.name = "output1_"; 
output1_.ref_num = 1;
output1_.value = 0.000000;
output1_.prev = &prevNodesoutput1_;
output1_.c = 2;
output1_.curr_c = 2;
double actParamsoutput1_[2] ={1.000000,0};
output1_.params = actParamsoutput1_;
output1_.n = 2;
output1_.visited_front = 1;
output1_.act_func = linear;


node loss1_;
node* prevNodesloss1_[1] ={&output1_};
loss1_.name = "loss1_"; 
loss1_.ref_num = 0;
loss1_.value = 0.000000;
loss1_.prev = &prevNodesloss1_;
loss1_.c = 1;
loss1_.curr_c = 1;
double actParamsloss1_[1] ={0};
loss1_.params = actParamsloss1_;
loss1_.n = 1;
loss1_.visited_front = 1;
loss1_.act_func = mse;


c_graph c_gp;
node* roots[2] ={&loss0_,&loss1_};
c_gp.root = roots;
}