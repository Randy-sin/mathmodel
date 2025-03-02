import pandas as pd
import os

def process_patronage_data(file_path):
    """
    处理乘客数据，提取日期和访客数据
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：找不到文件 {file_path}")
            print("当前工作目录：", os.getcwd())
            print("目录内容：", os.listdir())
            return None
            
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 只保留日期和访客人数列
        result_df = df[['日期', '访客人数']].copy()
        
        # 将日期列转换为datetime格式
        result_df['日期'] = pd.to_datetime(result_df['日期'])
        
        # 格式化日期为YYYY-MM格式
        result_df['日期'] = result_df['日期'].dt.strftime('%Y-%m')
        
        # 重命名列
        result_df.columns = ['date', 'visitors']
        
        # 按日期排序
        result_df = result_df.sort_values('date')
        
        # 保存处理后的数据
        output_file = os.path.join('投放资源', 'processed_visitors.csv')
        result_df.to_csv(output_file, index=False)
        print(f"✓ 数据处理完成，已保存至 {output_file}")
        
        return result_df
        
    except Exception as e:
        print(f"处理数据时出错：{str(e)}")
        return None

if __name__ == "__main__":
    # 处理数据
    file_path = os.path.join('投放资源', 'combined_monthly_data.csv')
    df = process_patronage_data(file_path)
    if df is not None:
        print("\n数据预览：")
        print(df.head()) 