require 'webrick'
require 'webrick/httpproxy'

class SimpleHTTPServer
  def initialize(port)
    @server = TCPServer.new('127.0.0.1', port)
    @routes = {}
  end

  def mount_proc(path, proc = nil, &block)
    if proc
      @routes[path] = proc
    else
      @routes[path] = block
    end
  end

  def start
    @thread = Thread.new do
      loop do
        client = @server.accept
        request_line = client.gets
        headers = {}
        while (line = client.gets) && (line != "\r\n")
          key, value = line.chomp.split(/:\s*/, 2)
          headers[key] = value
        end
        next unless request_line

        method, path, _ = request_line.split(' ')
        handle_request(client, method, path, headers)
      end
    end
  end

  def shutdown
    @thread.kill
    @server.close
  end

  private

  def handle_request(client, method, path, headers)
    if @routes.key?(path)
      proc = @routes[path]
      req = Struct.new(:request_method, :path_info, :headers).new(method, path, headers)
      res = Struct.new(:body, :status).new("", 200)
      proc.call(req, res)
      client.print "HTTP/1.1 #{res.status}\r\nContent-Type: text/plain\r\n\r\n#{res.body}"
    else
      client.print "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot Found"
    end
  ensure
    client.close
  end
end

class SimpleHTTPProxyServer
  def initialize(port, auth_proc = nil)
    @server = TCPServer.new('127.0.0.1', port)
    @auth_proc = auth_proc
  end

  def start
    @thread = Thread.new do
      loop do
        client = @server.accept
        request_line = client.gets
        headers = {}
        while (line = client.gets) && (line != "\r\n")
          key, value = line.chomp.split(/:\s*/, 2)
          headers[key] = value
        end
        next unless request_line

        method, path, _ = request_line.split(' ')
        handle_request(client, method, path, request_line, headers)
      end
    end
  end

  def shutdown
    @thread.kill
    @server.close
  end

  private

  def handle_request(client, method, path, request_line, headers)
    if @auth_proc
      req = Struct.new(:request_method, :path_info, :request_line, :headers).new(method, path, request_line, headers)
      res = Struct.new(:body, :status).new("", 200)
      @auth_proc.call(req, res)
      if res.status != 200
        client.print "HTTP/1.1 #{res.status}\r\nContent-Type: text/plain\r\n\r\n#{res.body}"
        return
      end
    end

    uri = URI(path)
    proxy_request(uri, client)
  ensure
    client.close
  end

  def proxy_request(uri, client)
    Net::HTTP.start(uri.host, uri.port) do |http|
      response = http.get(uri.path)
      client.print "HTTP/1.1 #{response.code}\r\nContent-Type: #{response.content_type}\r\n\r\n#{response.body}"
    end
  end
end
