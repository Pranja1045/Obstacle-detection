
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class StreamChecker:
    def __init__(self, words):
        self.root = TrieNode()
        self.stream = []

        # Insert reversed words into the Trie
        for word in words:
            node = self.root
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True

    def query(self, letter):
        self.stream.append(letter)
        node = self.root

        # Traverse the stream in reverse
        for char in reversed(self.stream):
            if char not in node.children:
                return False
            node = node.children[char]
            if node.is_end:
                return True
        return False

if __name__ == "__main__":
    n=int(input())
    A=list(map(str,input().split()[:n]))
    n1=int(input())
    B=list(map(str,input().split()[:n1]))
    stream_checker = StreamChecker(A)
    for letter in B:
        print(stream_checker.query(letter),end=" ")


