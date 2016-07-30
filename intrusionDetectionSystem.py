import pandas as pd
import numpy as np
import math


class decision_tree:
	split_key = None 
	children = [None for i in range(0,10000)]
	entropy_of_split = 0.0
	gain_of_split = 0.0
	isLeaf = False
	prev_split_value = None 
	type_of_attack = None
	num_of_children = 0
	def __init__(self , key):
		self.split_key = key
	
	def set_split_key(self , key):
		self.split_key = key 
	
	def fill_node(self , key , entropy_of_split , gain_of_split , isLeaf ,type_of_attack,num_of_children,prev_split_value):
		self.prev_split_value = prev_split_value
		self.num_of_children = num_of_children
		self.split_key = key 
		self.entropy_of_split = entropy_of_split
		self.gain_of_split = gain_of_split
		self.type_of_attack = type_of_attack
		self.num_of_children = num_of_children


this_node = None 
df = None
df_new = None

def import_data(): 
   location = r'kddcup.data_10_percent_corrected'
   df = pd.read_csv(location)
   return df

def entropy_split(df):
   temp = df.groupby(['result'])
   entropy_s = 0.0
   p = 0.0
   freq_table = temp.size()
   sum1 = freq_table.sum()
   cnt = freq_table.count()
   for i in range(0,cnt):
   		p = (freq_table[i]*1.0) / sum1
   		entropy_s -= (1.0 * p * math.log(p,2))
   return entropy_s

ret = None 
def entropy_sub(key, att, df):
   temp = df.groupby([att, 'result'])
   freq = temp.size()
   # print freq
   freq = freq[key]
   sum = freq.sum()
   i = 0
   sum = 1.0 * sum
   randomness = 0
   for k,l in freq.iteritems():
   	# print freq[k]
   	p = l / sum 
   	randomness -= (p * math.log(p,2))
   	# print k,i
   	i+=1
   return randomness, sum

def infoGain(att, df, ent):
   sub_entropy = 0.0
   gain = ent
   temp = df.groupby([att])
   freq = temp.size()
   bigsum = 1.0 * freq.sum()
   # print 'total = ' , bigsum
   tot = 0.0
   att_cnt = 0 
   # print 'initial gain = ' , gain
   for key,label in freq.iteritems():    	       
       att_cnt += 1
       # print att , att_cnt
       sub_entropy, sum = entropy_sub(key, att, df)
       # print 'sub_entropy is ' , sub_entropy , ' cnt is ' ,sum
       gain -= (1.0 * (sum / bigsum) * sub_entropy)
       tot += sum
   # assert 0 
   assert tot == bigsum
   # print 'maxgain = ',gain
   return gain

def preProcess(df, attributes):
	for key in attributes:
		if attributes[key] == 'continuous':
			try:
				df[key] = pd.qcut(df[key].values , 5).codes
			except ValueError:
				try:
					df[key] = pd.qcut(df[key].values , 4).codes
				except ValueError:
					try:
						df[key] = pd.qcut(df[key].values , 3).codes
					except ValueError:
						try:
							df[key] = pd.qcut(df[key].values , 2).codes
						except ValueError:
							try:
								df[key] = pd.qcut(df[key].values , 1).codes
							except ValueError:
								continue

def split_data_frame(df ,att):
	temp = df.groupby(att)
	freq=temp.size()
	i=0
	df.sort(columns=[att], inplace=True)
	df.set_index(keys=[att], drop=False, inplace=True)
	# create unique list of names
	unique_attributes = df[att].unique()
	if att == 'service':
		print unique_attributes
	# #create a data frame dictionary to store your data frames
	dff = {elem : pd.DataFrame for elem in unique_attributes}
	for k,v in freq.iteritems():
		dff[i]=df.loc[df[att]==k]
		dff[i].drop(att,axis=1, inplace=True)
		i += 1

	return dff , i , unique_attributes

def classify(feature_v,root , df):
	temp_node = root 
	this_split_value = ' '
	flag = 0  
	while temp_node.isLeaf == False:
		this_split_key = temp_node.split_key
		# assert this_node.num_of_children > 0
		flag = 0
		for i in range(0 , temp_node.num_of_children):
			if temp_node.children[i].prev_split_value  == feature_v[this_split_key]:
				temp_node = temp_node.children[i]
				flag = 1 
				break 
		if flag == 0:
			return ret
		
	return temp_node.type_of_attack	


def makeDT(isTaken, df , root , prev_value):  # attributes[] store which attributes we have taken into consideration
   splitKey , max_gain , entropy , total_free_keys_considered = select_best_attribute(df,isTaken)
   
   if entropy == 0.0:
   	root.fill_node(None , 0.0 , 0.0 , True , df['result'].iloc[0] , 0 , prev_value)
   	return root
   	
   if splitKey == None:
	unique_attributes = df['result'].unique()
   	temp = df.groupby(['result'])
   	freq_table = temp.size()
   	key_val_max = freq_table.max()	
   	lab_val_max = ''
   	for key,val in freq_table.iteritems():
   		if val == key_val_max:
   			lab_val_max = key
   	
   	root.fill_node(None , 0.0 , 0.0 , True , lab_val_max , 0 , prev_value)
   	return root		

   splitted_data , cnt , unique_attributes = split_data_frame(df , splitKey)
   
   root.fill_node(splitKey , entropy , max_gain , False , None , cnt , prev_value)
   print 'root prev_split_value = ' , root.prev_split_value
   for i in range(0,cnt):
   	root.children[i] = decision_tree(None)
   	 		   
   if not df.empty:
   	print 'will assign children = ' , cnt
   	for i in range(0,cnt):		
   		root.children[i] = makeDT(isTaken , splitted_data[i] , root.children[i] , unique_attributes[i])
   		print unique_attributes[i], root.children[i].prev_split_value, i, splitKey	
   return root

def select_best_attribute(df , isTaken):
	ret = None
	entropy = 0.0
	gain = 0.0
	max_gain = 0.0 
	cnt = 0 
	entropy = entropy_split(df)
	# print 'entropy is ## ' , entropy
	if entropy == 0.0:
		return ret,max_gain,entropy,0
	for key in isTaken:
		if isTaken[key] == 0:
			# print 'key is ' , key , ' is it taken ' , isTaken[key]
			cnt += 1
			# print '@@ ' , key
			gain = infoGain(key , df , entropy)
			# print '$$ ',gain
			if gain >= max_gain:
				ret = key 
				max_gain=gain
	# if cnt != 0:
	isTaken[ret] = 1 
	return ret,max_gain,entropy,cnt


#main
# df = import_data()

attribute_types = {'duration': 'continuous',
'protocol_type' : 'symbolic',
'service':'symbolic',
'flag': 'symbolic',
'src_bytes': 'continuous',
'dst_bytes': 'continuous',
'land': 'symbolic',
'wrong_fragment': 'continuous',
'urgent': 'continuous',
'num_failed_logins': 'continuous',
'logged_in': 'symbolic',
'root_shell': 'continuous',
'su_attempted': 'continuous',
'num_root': 'continuous',
'num_file_creations': 'continuous',
'num_shells': 'continuous',
'num_access_files': 'continuous',
'num_outbound_cmds': 'continuous',
'is_host_login': 'symbolic',
'is_guest_login': 'symbolic',
'count': 'continuous',
'srv_count': 'continuous',
'serror_rate': 'continuous',
'srv_serror_rate': 'continuous',
'rerror_rate': 'continuous',
'srv_rerror_rate': 'continuous',
'same_srv_rate': 'continuous',
'diff_srv_rate': 'continuous',
'srv_diff_host_rate': 'continuous',
'dst_host_count': 'continuous',
'dst_host_srv_count': 'continuous',
'dst_host_same_srv_rate': 'continuous',
'dst_host_diff_srv_rate': 'continuous',
'dst_host_same_src_port_rate': 'continuous',
'dst_host_srv_diff_host_rate': 'continuous',
'dst_host_serror_rate': 'continuous',
'dst_host_srv_serror_rate': 'continuous',
'dst_host_rerror_rate': 'continuous',
'dst_host_srv_rerror_rate': 'continuous'};

#make a boolean array to keep track 

isTaken = {'duration': 0,
'protocol_type' : 0,
'service':0,
'flag': 0,
'src_bytes': 0,
'dst_bytes': 0,
'land': 0,
'wrong_fragment': 0,
'urgent': 0,
'num_failed_logins': 0,
'logged_in': 0,
'root_shell': 0,
'su_attempted': 0,
'num_root': 0,
'num_file_creations': 0,
'num_shells': 0,
'num_access_files': 0,
'num_outbound_cmds': 0,
'is_host_login': 0,
'is_guest_login': 0,
'count': 0,
'srv_count': 0,
'serror_rate': 0,
'srv_serror_rate': 0,
'rerror_rate': 0,
'srv_rerror_rate': 0,
'same_srv_rate': 0,
'diff_srv_rate': 0,
'srv_diff_host_rate': 0,
'dst_host_count': 0,
'dst_host_srv_count': 0,
'dst_host_same_srv_rate': 0,
'dst_host_diff_srv_rate': 0,
'dst_host_same_src_port_rate': 0,
'dst_host_srv_diff_host_rate': 0,
'dst_host_serror_rate': 0,
'dst_host_srv_serror_rate': 0,
'dst_host_rerror_rate': 0,
'dst_host_srv_rerror_rate': 0
};


df = import_data()

preProcess(df , attribute_types)

loc = r'test_data.data.corrected'
df_new = pd.read_csv(loc)
ret = df_new['result'].iloc[0]
counter = df_new['duration'].count()

df_new = df_new.append(df , ignore_index = False)



root = decision_tree(' ')
root = makeDT(isTaken , df , root , None)
for i in range(0,counter):
	print classify(df_new.iloc[i] , root , df)

