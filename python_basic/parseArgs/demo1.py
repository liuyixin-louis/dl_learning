import argparse

if __name__ == "__main__":
    # title - 输出帮助的子解析器分组的标题；如果提供了描述则默认为 "subcommands"，否则使用位置参数的标题
    # description - 输出帮助中对子解析器的描述，默认为 None
    # prog - 将与子命令帮助一同显示的用法信息，默认为程序名称和子解析器参数之前的任何位置参数。
    # parser_class - 将被用于创建子解析器实例的类，默认为当前解析器类（例如 ArgumentParser）
    # action - 当此参数在命令行中出现时要执行动作的基本类型
    # dest - 将被用于保存子命令名称的属性名；默认为 None 即不保存任何值
    # required - 是否必须要提供子命令，默认为 False (在 3.7 中新增)
    # help - 在输出帮助中的子解析器分组帮助信息，默认为 None
    # metavar - 帮助信息中表示可用子命令的字符串；默认为 None 并以 {cmd1, cmd2, ..} 的形式表示子命令

    parser = argparse.ArgumentParser(description="this is a description")
    parser.add_argument('--argu1',
    type=str,
    metavar='ar1', #help information
    dest = 'a1',#the varibale name of argu1
    help='help information of argu1')


    # 分组
    group1 = parser.add_argument_group('group1')
    group1.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')



    args = parser.parse_args() #将参数字符串转换为对象并将其设为命名空间的属性。 返回带有成员的命名空间。
    print(args.a1)


    

