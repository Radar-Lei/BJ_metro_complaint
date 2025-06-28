#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北京地铁投诉数据处理脚本
使用LM Studio本地大模型从工单内容中提取结构化信息
Author: 雷达
Contact: greatradar@gmail.com; dalei@cuhk.edu.hk; dalei@scu.edu.cn
"""

import pandas as pd
import lmstudio as lms
import time
from typing import Dict, Optional
from tqdm import tqdm
from pydantic import BaseModel

class ComplaintInfoSchema(BaseModel):
    """投诉信息提取结果的数据模型"""
    line: Optional[str] = None  # 线路
    location: Optional[str] = None  # 小区/位置
    noise_type: Optional[str] = None  # 噪音类型
    vibration_type: Optional[str] = None  # 振动类型

class MetroComplaintProcessor:
    def __init__(self, model_name: str = "qwen/qwen3-30b-a3b-mlx"):
        """
        初始化处理器
        
        Args:
            model_name: LM Studio中的模型名称
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化LM Studio模型"""
        try:
            print(f"正在连接LM Studio模型: {self.model_name}")
            self.model = lms.llm()
            print("模型连接成功!")
        except Exception as e:
            print(f"连接模型失败: {e}")
            raise
    
    def extract_info_from_content(self, content: str) -> Dict[str, Optional[str]]:
        """
        从工单内容中提取信息
        
        Args:
            content: 工单内容
            
        Returns:
            包含提取信息的字典
        """
        if pd.isna(content) or not content.strip():
            return {
                "线路": None,
                "小区/位置": None,
                "噪音类型": None,
                "振动类型": None
            }
        
        # 构建提示词
        prompt = f"""你是一个数据提取专家。请分析北京地铁投诉工单内容，提取关键信息。

工单内容：{content}

提取信息：
1. line: 地铁线路（如1号线、2号线、13号线等）
2. location: 小区名称、地点或站点
3. noise_type: 噪音问题类型（如列车噪音、施工噪音、机械噪音等）
4. vibration_type: 振动问题类型（如列车振动、施工振动等）

没有明确提及的信息返回null。"""
        
        try:
            # 使用结构化响应调用模型
            response = self.model.respond(prompt, response_format=ComplaintInfoSchema)
            
            # 直接从parsed字段获取结构化结果
            result = response.parsed
            
            # 处理结果 - 检查是否为字典格式
            def convert_null(value):
                """将字符串'null'转换为None"""
                return None if value == "null" else value
            
            if isinstance(result, dict):
                return {
                    "线路": convert_null(result.get("line")),
                    "小区/位置": convert_null(result.get("location")),
                    "噪音类型": convert_null(result.get("noise_type")),
                    "振动类型": convert_null(result.get("vibration_type"))
                }
            else:
                # 如果是Pydantic对象
                return {
                    "线路": convert_null(result.line),
                    "小区/位置": convert_null(result.location),
                    "噪音类型": convert_null(result.noise_type),
                    "振动类型": convert_null(result.vibration_type)
                }
                
        except Exception as e:
            print(f"模型调用失败: {e}")
            return {
                "线路": None,
                "小区/位置": None,
                "噪音类型": None,
                "振动类型": None
            }
    
    def process_excel_file(self, input_file: str, output_file: str):
        """
        处理Excel文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        print(f"正在读取文件: {input_file}")
        df = pd.read_excel(input_file)
        
        print(f"共有 {len(df)} 条记录需要处理")
        
        # 创建结果DataFrame
        results = []
        
        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
            content = row.get('工单内容', '')
            phone = row.get('来电号码', '')
            
            # 提取信息
            extracted_info = self.extract_info_from_content(content)
            
            # 构建结果行
            result_row = {
                '工单内容': content,
                '来电号码': phone,
                '线路': extracted_info['线路'],
                '小区/位置': extracted_info['小区/位置'],
                '噪音类型': extracted_info['噪音类型'],
                '振动类型': extracted_info['振动类型']
            }
            
            results.append(result_row)
            
            # 每处理10条记录稍作停顿，避免对模型造成过大压力
            if (idx + 1) % 10 == 0:
                time.sleep(1)
                print(f"已处理 {idx + 1} 条记录")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 保存到Excel文件
        print(f"正在保存结果到: {output_file}")
        result_df.to_excel(output_file, index=False)
        
        print("处理完成!")
        print(f"结果已保存到: {output_file}")
        
        # 显示统计信息
        self._print_statistics(result_df)
    
    def _print_statistics(self, df: pd.DataFrame):
        """打印统计信息"""
        print("\n=== 处理统计 ===")
        print(f"总记录数: {len(df)}")
        print(f"提取到线路信息的记录数: {df['线路'].notna().sum()}")
        print(f"提取到小区/位置信息的记录数: {df['小区/位置'].notna().sum()}")
        print(f"提取到噪音类型的记录数: {df['噪音类型'].notna().sum()}")
        print(f"提取到振动类型的记录数: {df['振动类型'].notna().sum()}")
        
        # 显示线路分布
        if df['线路'].notna().sum() > 0:
            print("\n线路分布:")
            line_counts = df['线路'].value_counts()
            for line, count in line_counts.head(10).items():
                print(f"  {line}: {count} 条")
        
        # 显示噪音类型分布
        if df['噪音类型'].notna().sum() > 0:
            print("\n噪音类型分布:")
            noise_counts = df['噪音类型'].value_counts()
            for noise_type, count in noise_counts.head(10).items():
                print(f"  {noise_type}: {count} 条")

def main():
    """主函数"""
    input_file = "data/2024全年表振动噪音-筛选_增强版2.xlsx"
    output_file = "data/处理结果_地铁投诉分析.xlsx"
    
    try:
        # 创建处理器
        processor = MetroComplaintProcessor()
        
        # 处理文件
        processor.process_excel_file(input_file, output_file)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 