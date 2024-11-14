import os


predicate_str = '''(cooked ?obj1)
(frozen ?obj1)
(open ?obj1)
(folded ?obj1)
(unfolded ?obj1)
(toggled_on ?obj1)
(hot ?obj1)
(on_fire ?obj1)
(future ?obj1)
(real ?obj1)
(saturated ?obj1 ?obj2)
(covered ?obj1 ?obj2)
(filled ?obj1 ?obj2)
(contains ?obj1 ?obj2)
(ontop ?obj1 ?obj2)
(nextto ?obj1 ?obj2)
(under ?obj1 ?obj2)
(touching ?obj1 ?obj2)
(inside ?obj1 ?obj2)
(overlaid ?obj1 ?obj2)
(attached ?obj1 ?obj2)
(draped ?obj1 ?obj2)
(insource ?obj1 ?obj2)
(inroom ?obj1 ?obj2)
(broken ?obj1)
(grasped ?obj1 ?obj2)'''

predicate_list = predicate_str.split('\n')

action_str_all = ''
for predicate in predicate_list:
    tuple = predicate[1:-1].split(' ')
    name = tuple[0]
    args = tuple[1:]
    arg_str = ' '.join(args)
    action_str = f'''
    (:action do_{name}
        :parameters ({arg_str})
        :precondition (not ({name} {arg_str}))
        :effect ({name} {arg_str})
    )

    (:action undo_{name}
        :parameters ({arg_str})
        :precondition ({name} {arg_str})
        :effect (not ({name} {arg_str}))
    )
'''
    action_str_all += action_str


with open(os.path.dirname(__file__) + '/action_str.txt', 'w') as f:
    f.write(action_str_all)