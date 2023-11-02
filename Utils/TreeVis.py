import json
import numpy as np
import pandas

def convert(bdt):
    bst = bdt.get_booster()
    meta = json.loads(bst.save_config())
    updater = meta.get('learner').get('gradient_booster').get('gbtree_train_param').get('updater').split(',')[0]
    #max_depth = int(meta.get('learner').get('gradient_booster').get('updater').get(updater).get('train_param').get('max_depth'))
    max_depth = int(meta.get('learner').get('gradient_booster').get('tree_train_param').get('max_depth'))
    n_classes = int(meta.get('learner').get('learner_model_param').get('num_class'))
    fn_classes = 1 if n_classes == 0 else n_classes # the number of learners
    n_classes = 2 if n_classes == 0 else n_classes # the actual number of classes
    n_trees = int(int(meta.get('learner').get('gradient_booster').get('gbtree_model_param').get('num_trees')) / fn_classes)
    n_features = int(meta['learner']['learner_model_param']['num_feature'])
    ensembleDict = {'max_depth' : max_depth,
                    'n_trees' : n_trees,
                    'n_classes' : n_classes,
                    'n_features' : n_features,
                    'trees' : [],
                    'init_predict' : [0] * n_classes,
                    'norm' : 1}
    
    feature_names = {}
    if bst.feature_names is None:
      for i in range(n_features):
        feature_names[f'f{i}'] = i
    else:
      for i, feature_name in enumerate(bst.feature_names):
        feature_names[feature_name] = i

    trees = bst.trees_to_dataframe()
    for i in range(ensembleDict['n_trees']):
        treesl = []
        for j in range(fn_classes):
            tree = trees[trees.Tree == fn_classes * i + j]
            tree = treeToDict(tree, feature_names)
            treesl.append(tree)
        ensembleDict['trees'].append(treesl)
    return ensembleDict
    
def treeToDict(tree : pandas.DataFrame, feature_names):
  assert isinstance(tree, pandas.DataFrame), "This method expects the tree as a pandas DataFrame"
  
  thresholds = tree.Split.fillna(0).tolist()
  IDs = tree.ID.map(lambda x : int(x.split('-')[1]) if isinstance(x, str) else -1).tolist()
  differences = [IDs[i+1]-IDs[i] for i in range(len(IDs)-1)]

  diff_idx = [ n for n,i in enumerate(differences) if i>1 ]

  if len(diff_idx) > 0:
     differences = [0 if i < diff_idx[0] else differences[diff_idx[0]] for i in range(len(differences))]
     differences.append(differences[diff_idx[0]])
  #print(IDs - differences)
  #print([IDs[i] - differences[i] for i in range(differences)])
  features = tree.Feature.map(lambda x : -2 if x == 'Leaf' else feature_names[x]).tolist()
  children_left = tree.Yes.map(lambda x : int(x.split('-')[1]) if isinstance(x, str) else -1).tolist()
  children_right = tree.No.map(lambda x : int(x.split('-')[1]) if isinstance(x, str) else -1).tolist()
  #print(differences)
  
  if (max(differences) > 1):
    #print(tree[['Tree','Node','ID','Feature','Split','Yes','No','Gain']])
    #print(children_left)
    #print(children_right)
    children_left = [children_left[i] - differences[i] if children_left[i] > 0 else children_left[i] for i in range(len(children_left))]
    children_right = [children_right[i] - differences[i] if children_right[i] > 0 else children_right[i] for i in range(len(children_right))]
    #print(children_left)
    #print(children_right)


  values = tree.Gain.tolist()
  treeDict = {'feature'        : features,
              'threshold'      : thresholds,
              'children_left'  : children_left,
              'children_right' : children_right,
              'value'          : values
              }
  return treeDict

def evaluateTree(X,tree,feature_list):
  i = 0;
  while(tree['feature'][i] != -2):
      comparison = X[feature_list[tree['feature'][i]]].values[0] <= tree['threshold'][i];
      if comparison:
        i = tree['children_left'][i]
      else:
        i = tree['children_right'][i];
  return tree['value'][i];