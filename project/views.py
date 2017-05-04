from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Context, Template
import string, operator, re, twitter
from collections import Counter, defaultdict
from wordcloud import WordCloud
from django.core.exceptions import *
from .forms import UserName
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


user_handle = ''
mon = tue = wed = thu = fri = sat = sun = 0
api = twitter.Api(consumer_key='*************************************',
	                  consumer_secret='*************************************',
                      access_token_key='*************************************',
                      access_token_secret='*************************************')
def rotate_keys():
	consumer_key1='*************************************'
	consumer_secret1='*************************************'
	access_token_key1='*************************************'
	access_token_secret1='*************************************'
	global api
	api = twitter.Api(consumer_key=consumer_key1,
	                  consumer_secret=consumer_secret1,
                      access_token_key=access_token_key1,
                      access_token_secret=access_token_secret1)
	
def rescale_arr(arr, amin, amax):
    m = arr.min()
    M = arr.max()
    if(M == m):
    	s = float(amax-amin)
    else:	
    	s = float(amax-amin)/(M-m)
    d = amin - s*m
    return np.clip(s*arr+d,amin,amax)

def co_occurrences_graph(word_hist, co_occur, cutoff=0):
	print("Creating co-occurrence graph......")
	g = nx.Graph()
	for word in word_hist:
		g.add_node(word)
	for (w1, w2), count in co_occur:
		if count <= cutoff:
			continue
		g.add_edge(w1, w2, weight=count)
	return g
rad0 = 0.3

def centrality_layout(wgraph, centrality):
	print("Generating centrality layout.............")
	cent = sorted(centrality.items(), key=lambda x:float(x[1]), reverse=True)
	nodes = [c[0] for c in cent]
	cent  = np.array([float(c[1]) for c in cent])
	rad = (cent - cent[0])/(cent[-1]-cent[0])
	rad = rescale_arr(rad, rad0, 1)
	angles = np.linspace(0, 2*np.pi, len(centrality))
	layout = {}
	for n, node in enumerate(nodes):
		r = rad[n]
		th = angles[n]
		layout[node] = r*np.cos(th), r*np.sin(th)
	return layout


def plot_graph(wgraph, pos=None, fig=None, title=None):
	print("Plotting graph............")
	edge_min_width= 3
	edge_max_width= 12
	label_font = 18
	node_font = 22
	node_alpha = 0.4
	edge_alpha = 0.55
	edge_cmap = plt.cm.Spectral
	if fig is None:
		fig, ax = plt.subplots()
	else:
		ax = fig.add_subplot(111)
	fig.subplots_adjust(0,0,1)
	degrees = []
	for n, d in wgraph.nodes(data=True):
		degrees.append(wgraph.degree(n))
	pos = nx.spring_layout(wgraph) if pos is None else pos
	labels = {}
	width = []
	for n1, n2, d in wgraph.edges_iter(data=True):
		w = d['weight']
		labels[n1, n2] = w
		width.append(w)
	width = rescale_arr(np.array(width, dtype=float), edge_min_width, edge_max_width)
	nx.draw_networkx_labels(wgraph, pos, font_size=node_font, font_weight='bold')
	nx.draw_networkx_edge_labels(wgraph, pos, edge_labels=labels, font_size=label_font)
	nx.draw_networkx_nodes(wgraph, pos, node_size=degrees, node_color='red',alpha=node_alpha)
	nx.draw_networkx_edges(wgraph, pos, width=width, edge_color=width,edge_cmap=edge_cmap, alpha=edge_alpha)
	plt.tight_layout()
	plt.savefig('./project/static/project/images/myfig.png',dpi = 1000)
	plt.clf()
	
#Does not work for wrong user_handles
def get_tweets():
	print("Gathering tweets....")
	alltweets = []
	new_tweets = api.GetUserTimeline(screen_name= user_handle, exclude_replies=True, include_rts=False, count = 200)
	alltweets.extend(new_tweets)
	oldest = alltweets[-1].id - 1
	while len(new_tweets) > 0:
		new_tweets = api.GetUserTimeline(screen_name = user_handle , include_rts = False, exclude_replies = True, count=200 , max_id=oldest)
		alltweets.extend(new_tweets)
		oldest = alltweets[-1].id - 1
	return alltweets

def get_name(request):
	if request.method == 'POST':
		form = UserName(request.POST)
		if form.is_valid():
			global user_handle
			user_handle = form.cleaned_data['user_name']
			data = get_tweets()
			lists = []
			lis = []
			dates = []
			print("Starting the analysis....")
			for tweet in data:
				dates.append(tweet.created_at)
				lists.append(tweet.text)
			for date in dates:
				splitted = date.split()
				if(splitted[0] == "Sun"):
					global sun
					sun += 1;
				elif(splitted[0] == "Mon"):
					global mon
					mon += 1;
				elif(splitted[0] == "Tue"):
					global tue
					tue += 1;
				elif(splitted[0] == "Wed"):
					global wed
					wed += 1;
				elif(splitted[0] == "Thu"):
					global thu
					thu += 1;
				elif(splitted[0] == "Fri"):
					global fri
					fri += 1;
				elif(splitted[0] == "Sat"):
					global sat
					sat += 1;
			content_sublists = [line.split(',') for line in lists]
			content_list = [item for sublist in content_sublists for item in sublist]
			content_list_strip = [str.strip().lower() for str in content_list]
			content_concat = ' '.join(content_list_strip)
			punct = set(string.punctuation)
			unpunct_content = ''.join(x for x in content_concat if x not in punct)
			word_list = unpunct_content.split()
			regex = '^http'
			i = 0
			for word in word_list:
				if re.match(regex,word) is not None:
					del word_list[i]
				i += 1
			i = 0
			for word in word_list:
				if len(word) < 4:
					del word_list[i]
				i += 1
			print("Counting words...........")
			counts_all = Counter(word_list)
			words, count_values = zip(*counts_all.items())
			values_sorted, words_sorted = zip(*sorted(zip(count_values, words), key=operator.itemgetter(0), reverse=True))
			lis = list(zip(list(words_sorted),list(values_sorted)))
			com = defaultdict(lambda : defaultdict(int))
			print("Starting co-occurrence graph production.............")
			for i in range(len(word_list)-1):            
				for j in range(i+1, len(word_list)):
					w1, w2 = sorted([word_list[i], word_list[j]])                
					if w1 != w2:
						com[w1][w2] += 1
			com_max = []
			print("Gathering terms...............")
			for t1 in com:
			    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
			    for t2, t2_count in t1_max_terms:
			        com_max.append(((t1, t2), t2_count))
			terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
			test = terms_max[:10]
			sent = []
			for (w1,w2),count in test:				
				if(w1 not in sent):
					sent.append(w1)
				if(w2 not in sent):
					sent.append(w2)
			wordcloud = WordCloud(max_words = 100, max_font_size=100, font_path='./project/ReenieBeanie.ttf', relative_scaling=0.1).generate_from_frequencies(lis)
			image = wordcloud.to_image()
			image.save('./project/static/project/images/image.jpg')
			wgraph = co_occurrences_graph(sent, terms_max[:10], cutoff=1)
			wgraph = list(nx.connected_component_subgraphs(wgraph))[0]
			centrality = nx.eigenvector_centrality_numpy(wgraph)
			plot_graph(wgraph, centrality_layout(wgraph, centrality), plt.figure(figsize=(12,12)))

			print("Generating network graph.............")
			followers = api.GetFollowers(screen_name=user_handle)
			follower_name = [i.name for i in followers]
			friends = api.GetFriends(screen_name=user_handle)
			friend_name = [i.name for i in friends]
			followers_not_friends = set(follower_name).difference(friend_name)
			graph_list_followers = [(user_handle, name) for name in followers_not_friends]
			graph_list_friends = [(user_handle,name) for name in friend_name]
			G = nx.DiGraph()
			G.add_edges_from(graph_list_followers)
			G.add_edges_from(graph_list_friends)
			red_edges = graph_list_followers
			edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
			pos = nx.random_layout(G)
			nx.draw(G, pos, node_color='blue', node_size=50, font_size=15, edge_color=edge_colors, edge_cmap=plt.cm.Reds, with_labels=True)
			plt.savefig('./project/static/project/images/mygraph.png',dpi = 1000)
			return HttpResponseRedirect('result',user_handle)
	else:
		form = UserName()
	return render(request, 'project/search.html', {'form': form})


def index(request):
	html = ''
	template = loader.get_template('project/index.html')
	context = {
		'tweets' : get_tweets(),
		'user_handle' : user_handle,
		'sun' : sun,
		'mon' : mon,
		'tue' : tue,
		'wed' : wed,
		'thu' : thu,
		'fri' : fri,
		'sat' : sat,
	}
	return HttpResponse(template.render(context, request))