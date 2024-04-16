import re
import os
def remove_non_digits(input_string):
    # 使用正则表达式匹配非数字字符，并替换为空字符串
    return re.sub(r'\D', '', input_string)

def sort_by_numeric_value(strings):
    # 定义比较函数，按照去除非数字字符后的字符串转换为整数来比较大小
    def numeric_value(s):
        return int(remove_non_digits(s))

    # 使用sorted函数进行排序，并指定比较函数
    return sorted(strings, key=numeric_value)


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            else:
                print(f"Skipping directory: {file_path}")
        except Exception as e:
            print(f"Error: {e}")