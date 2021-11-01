with tqdm.tqdm(total=len(lines), leave=True, desc="preprocessing", position=0) as pbar:

            pbar.write("START")

            for line in lines:
                if not line: 
                    noItemNo += 1
                    pbar.set_description_str(f'no item at {i}', refresh=True)
                    sentences.append("")
                    i += 1
                    pbar.update(1)
                    continue

                if i % 500 == 0:
                    pbar.set_postfix_str("current(every 500) - " + str(i+1))
                i += 1