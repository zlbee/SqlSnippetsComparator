#!/usr/bin/env python
# coding: utf-8

# # SQL语句解析及比较工具

# In[1]:


import sys
print(sys.version)


# In[2]:


# pip install ipdb


# In[3]:


import difflib
import sqlparse
import pdb


# In[4]:


# 全局参数
CANDIDATES = 1 # 相似候选人数
THRESHOLD = 3 # 相似性指标裁剪阈值
NCOMPARE_FIELDS = ['created_at'] # 不参与比较的字段


# ## 解析

# In[5]:


# 冗余的token
redundant_tokens = {sqlparse.tokens.Whitespace, sqlparse.tokens.Punctuation}

def parse_insert_statement(sql):
    """解析插入语句"""
    # 解析SQL语句
    statement = sqlparse.parse(sql)[0]
    statement_tokens = statement.tokens
    
    cleaned_statement_tokens = [token for token in statement_tokens if token.ttype not in redundant_tokens]
    
    # 提取表名
    table_name = cleaned_statement_tokens[2].get_name()
    
    # 提取列名
    columns = []
    column_flag = False
    for child in cleaned_statement_tokens[2].flatten():
        if child.ttype == sqlparse.tokens.Name and column_flag:
            columns.append(child.value)
    
        if child.value == table_name:
            column_flag = True
    
    # 提取值
    values = []
    for child in cleaned_statement_tokens[3].flatten():
        if child.ttype not in redundant_tokens:
            values.append(child.value)
    values = values[1:]

    # 组合列名-值
    columns_values_map = dict(zip(columns, values))
    
    return {
        'table_name': table_name,
        'columns': columns,
        'values': values,
        'columns_values_map': columns_values_map
    }


# In[6]:


# 测试
sql = "INSERT INTO gl.your_table (id, name, age, created_at) VALUES (1, 'Alice', 30, '2025-01-26 10:00:00');"
result = parse_insert_statement(sql)
result


# ## 相似性计算

# In[7]:


def calc_similarity(parsed_tokens_left, parsed_tokens_right, ncompare_fields=NCOMPARE_FIELDS):
    """比对两组语句tokens的差异"""
    metric = 0
    parsed_tokens_left_copy = parsed_tokens_left.copy()
    parsed_tokens_right_copy = parsed_tokens_right.copy()
    
    # 比对表名
    if parsed_tokens_left_copy['table_name'] != parsed_tokens_right_copy['table_name']:
        metric = float('inf')
        return metric

    # 不比较字段定义
    for ncompare_field in ncompare_fields:
        if ncompare_field in parsed_tokens_left_copy['columns']:
            parsed_tokens_left_copy['columns_values_map'].pop(ncompare_field, None)
        if ncompare_field in parsed_tokens_right_copy['columns']:
            parsed_tokens_right_copy['columns_values_map'].pop(ncompare_field, None)

    # 字段名排序
    columns_values_map_left = dict(sorted(parsed_tokens_left_copy['columns_values_map'].items()))
    columns_values_map_right = dict(sorted(parsed_tokens_right_copy['columns_values_map'].items()))

    # 比对字段名及值
    for compare_item_left, compare_item_right in zip(columns_values_map_left.items(), columns_values_map_right.items()):
        # 比对字段名
        if compare_item_left[0] != compare_item_right[0]:
            metric += 1
        # 比对值
        if compare_item_left[1] != compare_item_right[1]:
            metric += 1

    return metric


# In[8]:


# 测试
sql1 = "INSERT INTO sungl.your_table (id, name, age, created_at) VALUES (1, 'Alice', 30, '2025-01-26 10:00:00');"
sql2 = "INSERT INTO sungl.your_table (id, name, age, created_at) VALUES (1, 'Bob', 30, '2025-01-27 10:00:00');"

parsed_sql1 = parse_insert_statement(sql1)
parsed_sql2 = parse_insert_statement(sql2)
metric = calc_similarity(parsed_sql1, parsed_sql2, ncompare_fields=['created_at'])
metric


# ## 比对

# In[9]:


def read_file(file_path):
    """读取文件内容并返回按行分割的列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_diff(result, output_file="diff_result.json"):
    import json
    """将差异结果写入文件"""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, default=serialize, ensure_ascii=False)


# In[10]:


def compare_files(file_left, file_right):
    """比对两个文件的差异"""
    # 读取文件内容并去除每行的换行符
    lines_left = [line.strip() for line in read_file(file_left)]
    lines_right = [line.strip() for line in read_file(file_right)]

    # 按升序排序
    lines_left.sort()
    lines_right.sort()

    # 找出差异行
    diff_left = set(lines_left) - set(lines_right)  # 在right中但不在right中
    diff_right = set(lines_right) - set(lines_left)  # 在right中但不在right中

    # 格式化输出差异
    diff_result = {}

    if diff_left:
        diff_result["left_uniques"] = []
        for line in sorted(diff_left):
            diff_result["left_uniques"].append(line)
    if diff_right:
        diff_result["right_uniques"] = []
        for line in sorted(diff_right):
            diff_result["right_uniques"].append(line)

    return diff_result, lines_left, lines_right


# In[11]:


diff_result, lines_left, lines_right = compare_files('./left.txt', './right.txt')
diff_result


# In[12]:


class SimilarityInfo:
    def __init__(self, original_line, target_line, metric, diff_detail):
        self.original_line = original_line
        self.target_line = target_line
        self.metric = metric
        self.diff_detail = diff_detail
        

    def __str__(self):
        diff_detail_modified = ''.join([str(line) + '\n' for line in self.diff_detail])
        # pdb.set_trace()
        return f"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
original_line:
{self.original_line}

target_line:
{self.target_line}

metric: 
{self.metric}

diff_detail: 
------------------------------------------
{diff_detail_modified}
------------------------------------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
        
def serialize(obj):
    if isinstance(obj, SimilarityInfo):
        return obj.__dict__  # 递归转换
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    

def generate_multi_similarities(original_lines, target_lines, candidates=CANDIDATES, threshold=THRESHOLD):
    """生成多行相似性"""
    # 使用 difflib 来比较两个字符串
    differ = difflib.Differ()
    metrics = {}

    
    for original_line in original_lines:
        # 解析原行
        original_line_parsed = parse_insert_statement(original_line)
    
        metrics_single_line = []
        for target_line in target_lines:
            # 解析目标行
            target_line_parsed = parse_insert_statement(target_line)
    
            # 计算指标
            metric = calc_similarity(original_line_parsed, target_line_parsed)

            # 计算差异
            diff_detail = list(differ.compare(original_line.splitlines(), target_line.splitlines()))
            
            # metrics_single_line.append([target_line, metric, diff])
            if metric <= threshold:
                metrics_single_line.append(SimilarityInfo(original_line, target_line, metric, diff_detail))
    
        # 排序
        metrics_single_line = sorted(metrics_single_line, key=lambda x: x.metric)
        
        # 截取候选人
        metrics_single_line = metrics_single_line[:candidates]
        
        metrics[original_line] = metrics_single_line
    return metrics


# In[13]:


if __name__ == '__main__':
    left_metrics = generate_multi_similarities(diff_result['left_uniques'], lines_right)
    
    for left_metric in left_metrics.items():
        print("================================================")
        print("ORIGINAL_LINE\n" + left_metric[0] + "\n")
        for left_metric_similarity_info in left_metric[1]:
            print(left_metric_similarity_info)
        print("================================================")
    
    
    # 写入文件
    write_diff(left_metrics, output_file='diff_result_left.json')


# In[14]:


if __name__ == '__main__':
    right_metrics = generate_multi_similarities(diff_result['right_uniques'], lines_left)
    for right_metric in right_metrics.items():
        print("================================================")
        print("ORIGINAL_LINE\n" + right_metric[0] + "\n")
        for right_metric_similarity_info in right_metric[1]:
            print(right_metric_similarity_info)
        print("================================================")
    
    # 写入文件
    write_diff(right_metrics, output_file='diff_result_right.json')

